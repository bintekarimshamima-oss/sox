"""UN Comtrade preview API ingestion for semiconductor trade flows.

Pulls monthly world-aggregate (partnerCode=0) import/export values for the
two HS families defined in config/sources.yml (integrated_circuits=8542,
semiconductor_equipment=8486) for each reporter in the reporters dict, over
the study window 2006-01 through the latest complete month before the
protocol freeze date.

Design notes (see task spec / config/protocol.yml):
  - One raw JSON snapshot per (reporter, hs_family, flow, period) call,
    written immediately and never overwritten, so partial progress survives
    interruption.
  - One manifest row per call appended to data/manifests/comtrade_manifest.jsonl
    as soon as the call completes (success or terminal failure), so the
    manifest always reflects exactly what has been attempted so far.
  - A resumption manifest (data/manifests/comtrade_resume_state.json) is
    rewritten after every call with the set of combos completed, so a future
    run can skip already-fetched combos without re-querying.
  - On repeated 429s that don't clear after long backoff, or on a schema
    change (no "data" key, HTML body, terms/access error), the script stops
    immediately and logs the exact failure -- it does not fabricate data or
    scrape around the wall.
"""

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw" / "comtrade"
MANIFEST_PATH = ROOT / "data" / "manifests" / "comtrade_manifest.jsonl"
RESUME_PATH = ROOT / "data" / "manifests" / "comtrade_resume_state.json"
INTERIM_DIR = ROOT / "data" / "interim"
LOG_DIR = ROOT / "outputs" / "logs"

HEADERS = {"User-Agent": "PhD Research Project yusha@iscb.org"}
STUDY_START = "2006-01"

# Freeze date is 2026-09-13 per config/protocol.yml -> latest *complete*
# month before that is 2026-08.
LATEST_COMPLETE_MONTH = "2026-08"

REQUEST_DELAY_SECONDS = 1.5
BACKOFF_SCHEDULE = [10, 30, 60]  # seconds, exponential-ish backoff on 429
MAX_CONSECUTIVE_HARD_FAILURES = 3  # combos, not requests -- see main loop


def load_sources():
    with open(ROOT / "config" / "sources.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def month_range(start_yyyymm: str, end_yyyymm: str):
    """Yield YYYYMM strings from start to end inclusive. Inputs are 'YYYY-MM'."""
    start_y, start_m = int(start_yyyymm.split("-")[0]), int(start_yyyymm.split("-")[1])
    end_y, end_m = int(end_yyyymm.split("-")[0]), int(end_yyyymm.split("-")[1])
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        yield f"{y:04d}{m:02d}"
        m += 1
        if m > 12:
            m = 1
            y += 1


def load_resume_state() -> set:
    """Return the set of 'reporter|hsfamily|flow|period' combos already
    successfully completed (HTTP 200, well-formed response, saved to disk)."""
    if RESUME_PATH.exists():
        with open(RESUME_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        return set(state.get("completed", []))
    return set()


def save_resume_state(completed: set, status: str, note: str = ""):
    RESUME_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,  # "complete" | "partial_stopped" | "in_progress"
        "note": note,
        "completed_count": len(completed),
        "completed": sorted(completed),
    }
    with open(RESUME_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def combo_key(reporter_iso: str, hs_family_code: str, flow: str, period: str) -> str:
    return f"{reporter_iso}|{hs_family_code}|{flow}|{period}"


def raw_path_for(reporter_iso: str, hs_family_code: str, flow: str, period: str) -> Path:
    return RAW_DIR / f"{reporter_iso}_{hs_family_code}_{flow}_{period}.json"


def append_manifest(record: dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


class SchemaChangedError(Exception):
    """Raised when the API response no longer matches the expected shape."""


class HardRateLimitError(Exception):
    """Raised when repeated 429s do not clear even after full backoff."""


def fetch_one(base_url: str, reporter_code: int, cmd_code: str, flow: str, period: str):
    """Perform one API call with exponential backoff on 429.

    Returns (response_text, http_status, url, params, elapsed_backoff_used).
    Raises HardRateLimitError if all backoff attempts are exhausted with 429.
    Raises SchemaChangedError if the response is not well-formed JSON with a
    'data' key (e.g. HTML body or a terms/access-restriction error payload).
    """
    params = {
        "reporterCode": reporter_code,
        "period": period,
        "partnerCode": 0,
        "cmdCode": cmd_code,
        "flowCode": flow,
    }

    attempt = 0
    while True:
        resp = httpx.get(base_url, params=params, timeout=30, headers=HEADERS)
        full_url = str(resp.url)

        if resp.status_code == 429:
            if attempt >= len(BACKOFF_SCHEDULE):
                raise HardRateLimitError(
                    f"429 persisted after {len(BACKOFF_SCHEDULE)} backoff attempts "
                    f"for {full_url}"
                )
            wait_s = BACKOFF_SCHEDULE[attempt]
            print(f"    429 received, backing off {wait_s}s (attempt {attempt + 1})...")
            time.sleep(wait_s)
            attempt += 1
            continue

        # Any other non-200: treat as a logged failure for this combo, not a
        # hard stop, unless the body indicates a schema/access change.
        content_type = resp.headers.get("content-type", "")
        text = resp.text

        if "html" in content_type.lower() or text.lstrip().lower().startswith("<!doctype") or text.lstrip().startswith("<html"):
            raise SchemaChangedError(
                f"HTML response received instead of JSON for {full_url} "
                f"(status {resp.status_code}). Body prefix: {text[:300]!r}"
            )

        if resp.status_code == 200:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as e:
                raise SchemaChangedError(
                    f"Non-JSON 200 response for {full_url}: {e}. Body prefix: {text[:300]!r}"
                )
            if "data" not in payload:
                # Could be a terms/access-restriction error payload.
                raise SchemaChangedError(
                    f"Response missing 'data' key for {full_url}. "
                    f"Body prefix: {text[:500]!r}"
                )

        return text, resp.status_code, full_url, dict(params)


def parse_row_value(row: dict, flow: str):
    """primaryValue falling back to cifvalue (imports) / fobvalue (exports)."""
    primary = row.get("primaryValue")
    if primary is not None:
        return primary
    if flow == "M":
        return row.get("cifvalue")
    if flow == "X":
        return row.get("fobvalue")
    return None


def run_ingestion():
    cfg = load_sources()
    ct_cfg = cfg["un_comtrade"]
    base_url = ct_cfg["base_preview"]
    hs_families = ct_cfg["hs_families"]  # name -> code
    reporters = ct_cfg["reporters"]  # iso -> numeric code
    flows = ct_cfg["flows"]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    periods = list(month_range(STUDY_START, LATEST_COMPLETE_MONTH))
    completed = load_resume_state()

    combos = []
    for reporter_iso, reporter_code in reporters.items():
        for hs_name, hs_code in hs_families.items():
            for flow in flows:
                for period in periods:
                    combos.append((reporter_iso, reporter_code, hs_name, hs_code, flow, period))

    total_planned = len(combos)
    already_done = sum(1 for c in combos if combo_key(c[0], c[3], c[4], c[5]) in completed)
    print(f"Planned combos: {total_planned}. Already completed (resume): {already_done}.")

    n_attempted_this_run = 0
    n_success_this_run = 0
    n_failed_this_run = 0
    failure_reasons = {}
    stop_reason = None

    for (reporter_iso, reporter_code, hs_name, hs_code, flow, period) in combos:
        key = combo_key(reporter_iso, hs_code, flow, period)
        if key in completed:
            continue

        out_path = raw_path_for(reporter_iso, hs_code, flow, period)
        if out_path.exists():
            # Raw snapshot already exists on disk but wasn't marked complete
            # (e.g. manifest write got interrupted). Never overwrite; treat
            # as already fetched and just resync resume-state + manifest gap.
            completed.add(key)
            continue

        n_attempted_this_run += 1
        retrieval_utc = datetime.now(timezone.utc).isoformat()

        try:
            text, status, url, params = fetch_one(base_url, reporter_code, hs_code, flow, period)
        except HardRateLimitError as e:
            stop_reason = f"hard_rate_limit: {e}"
            print(f"STOPPING: {stop_reason}")
            break
        except SchemaChangedError as e:
            stop_reason = f"schema_changed: {e}"
            print(f"STOPPING: {stop_reason}")
            break
        except httpx.HTTPError as e:
            n_failed_this_run += 1
            reason = f"network_error:{type(e).__name__}"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            append_manifest({
                "reporter": reporter_iso,
                "hs_family": hs_code,
                "flow": flow,
                "period": period,
                "url": base_url,
                "params": {
                    "reporterCode": reporter_code, "period": period,
                    "partnerCode": 0, "cmdCode": hs_code, "flowCode": flow,
                },
                "retrieval_utc": retrieval_utc,
                "http_status": None,
                "sha256": None,
                "row_count": None,
                "error": reason,
            })
            print(f"  FAILED (network) {reporter_iso}/{hs_code}/{flow}/{period}: {e}")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if status == 200:
            try:
                payload = json.loads(text)
                row_count = len(payload.get("data", []))
            except json.JSONDecodeError:
                row_count = None

            out_path.write_text(text, encoding="utf-8")

            append_manifest({
                "reporter": reporter_iso,
                "hs_family": hs_code,
                "flow": flow,
                "period": period,
                "url": url,
                "params": params,
                "retrieval_utc": retrieval_utc,
                "http_status": status,
                "sha256": sha256,
                "row_count": row_count,
            })
            completed.add(key)
            n_success_this_run += 1
            if n_success_this_run % 50 == 0:
                print(f"  ...{n_success_this_run} succeeded this run "
                      f"({len(completed)}/{total_planned} total)")
        else:
            n_failed_this_run += 1
            reason = f"http_{status}"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            append_manifest({
                "reporter": reporter_iso,
                "hs_family": hs_code,
                "flow": flow,
                "period": period,
                "url": url,
                "params": params,
                "retrieval_utc": retrieval_utc,
                "http_status": status,
                "sha256": sha256,
                "row_count": None,
                "error": reason,
            })
            print(f"  FAILED (http {status}) {reporter_iso}/{hs_code}/{flow}/{period}")

        # Periodically persist resume state so interruption never loses
        # more than one call's worth of progress.
        if (n_success_this_run + n_failed_this_run) % 25 == 0:
            save_resume_state(completed, status="in_progress")

        time.sleep(REQUEST_DELAY_SECONDS)

    final_status = "partial_stopped" if stop_reason else (
        "complete" if len(completed) == total_planned else "partial_stopped"
    )
    save_resume_state(completed, status=final_status, note=stop_reason or "")

    summary = {
        "total_planned": total_planned,
        "already_done_before_run": already_done,
        "attempted_this_run": n_attempted_this_run,
        "succeeded_this_run": n_success_this_run,
        "failed_this_run": n_failed_this_run,
        "failure_reasons": failure_reasons,
        "total_completed": len(completed),
        "stop_reason": stop_reason,
    }
    return summary, periods


def build_long_table():
    """Parse every saved raw JSON snapshot into one long-format table."""
    import pandas as pd

    rows = []
    for path in sorted(RAW_DIR.glob("*.json")):
        stem = path.stem  # <reporter>_<hsfamily>_<flow>_<period>
        parts = stem.split("_")
        if len(parts) != 4:
            print(f"  skipping unexpected filename: {path.name}")
            continue
        reporter_iso, hs_family, flow, period = parts

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  skipping unparseable JSON: {path.name}")
            continue

        data = payload.get("data", [])
        if not data:
            continue

        for row in data:
            trade_value = parse_row_value(row, flow)
            rows.append({
                "reporter_iso": reporter_iso,
                "reporter_code": row.get("reporterCode"),
                "hs_family": hs_family,
                "flow": flow,
                "period": period,
                "trade_value_usd": trade_value,
                "qty": row.get("qty"),
                "qty_unit": row.get("qtyUnitAbbr"),
                "net_weight": row.get("netWgt"),
                "is_reported": row.get("isReported"),
                "is_aggregate": row.get("isAggregate"),
            })

    df = pd.DataFrame(rows, columns=[
        "reporter_iso", "reporter_code", "hs_family", "flow", "period",
        "trade_value_usd", "qty", "qty_unit", "net_weight",
        "is_reported", "is_aggregate",
    ])

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DIR / "comtrade_trade_flows.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}")
    return df, out_path


def build_coverage_report(periods_attempted_by_combo: dict):
    """For each reporter x hs_family x flow: months retrieved (i.e. a raw
    JSON snapshot was successfully saved, regardless of whether that month
    had actual trade rows) out of months attempted, per protocol's
    ~70% rolling-coverage trust threshold.
    """
    import pandas as pd

    cfg = load_sources()
    ct_cfg = cfg["un_comtrade"]
    hs_families = ct_cfg["hs_families"]
    reporters = ct_cfg["reporters"]
    flows = ct_cfg["flows"]

    records = []
    for reporter_iso in reporters:
        for hs_name, hs_code in hs_families.items():
            for flow in flows:
                attempted = periods_attempted_by_combo.get((reporter_iso, hs_code, flow), [])
                n_attempted = len(attempted)
                n_retrieved = sum(
                    1 for period in attempted
                    if raw_path_for(reporter_iso, hs_code, flow, period).exists()
                )
                coverage_ratio = (n_retrieved / n_attempted) if n_attempted else 0.0
                records.append({
                    "reporter_iso": reporter_iso,
                    "hs_family": hs_code,
                    "hs_family_name": hs_name,
                    "flow": flow,
                    "months_attempted": n_attempted,
                    "months_retrieved": n_retrieved,
                    "coverage_ratio": round(coverage_ratio, 4),
                    "below_trust_threshold_0.70": coverage_ratio < 0.70,
                })

    df = pd.DataFrame(records)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DIR / "comtrade_coverage_report.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote coverage report -> {out_path}")
    return df, out_path


def write_status_log(summary: dict, coverage_df, periods: list):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / "comtrade_ingest_status.md"

    total_planned = summary["total_planned"]
    total_completed = summary["total_completed"]
    pct = (total_completed / total_planned * 100) if total_planned else 0.0

    lines = []
    lines.append("# UN Comtrade ingestion status")
    lines.append("")
    lines.append(f"Run completed (script exit) at: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Overall progress")
    lines.append("")
    lines.append(f"- Total combos planned (reporters x hs_families x flows x months, "
                  f"{STUDY_START} through {LATEST_COMPLETE_MONTH}): **{total_planned}**")
    lines.append(f"- Total combos completed (across all runs, cumulative): **{total_completed}** "
                  f"({pct:.1f}%)")
    lines.append(f"- Already done before this run: {summary['already_done_before_run']}")
    lines.append(f"- Attempted this run: {summary['attempted_this_run']}")
    lines.append(f"- Succeeded this run: {summary['succeeded_this_run']}")
    lines.append(f"- Failed this run: {summary['failed_this_run']}")
    lines.append("")
    if summary["failure_reasons"]:
        lines.append("### Failure reason breakdown (this run)")
        lines.append("")
        for reason, count in sorted(summary["failure_reasons"].items(), key=lambda kv: -kv[1]):
            lines.append(f"- {reason}: {count}")
        lines.append("")

    if summary["stop_reason"]:
        lines.append("## STOPPED EARLY")
        lines.append("")
        lines.append(f"Reason: `{summary['stop_reason']}`")
        lines.append("")
        lines.append(
            "The script halted per the guardrail against continuing past a hard "
            "access wall or a response-schema change. No data was fabricated for "
            "unretrieved months; they remain absent from the parquet table and "
            "show up as coverage gaps below."
        )
        lines.append("")
        lines.append("### Resumption instructions")
        lines.append("")
        lines.append(
            "1. Re-run `python -m soxstudy.ingest.comtrade` (or "
            "`python src/soxstudy/ingest/comtrade.py`) from the repo root.\n"
            "2. The script reads `data/manifests/comtrade_resume_state.json` and "
            "`data/raw/comtrade/*.json` on startup and automatically skips every "
            "(reporter, hs_family, flow, period) combo already completed -- it "
            "will only attempt the remaining combos.\n"
            "3. If the stop reason was a hard rate limit, wait at least "
            "several hours (ideally until the next UTC day, in case of a daily "
            "quota) before resuming.\n"
            "4. If the stop reason was `schema_changed`, inspect the offending "
            "URL manually in a browser first -- do not resume until the response "
            "shape is understood, since the parser assumes a stable schema."
        )
        lines.append("")
    else:
        lines.append("## Completed")
        lines.append("")
        lines.append("All planned combos were attempted in this run (cumulative across "
                      "all runs). See coverage report for per-cell completeness.")
        lines.append("")

    lines.append("## Coverage by reporter / hs_family / flow")
    lines.append("")
    if coverage_df is not None and len(coverage_df):
        lines.append("| reporter | hs_family | flow | attempted | retrieved | coverage | below 70% |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in coverage_df.iterrows():
            lines.append(
                f"| {r['reporter_iso']} | {r['hs_family']} | {r['flow']} | "
                f"{r['months_attempted']} | {r['months_retrieved']} | "
                f"{r['coverage_ratio']:.2%} | {'YES' if r['below_trust_threshold_0.70'] else 'no'} |"
            )
        lines.append("")
        n_below = int(coverage_df["below_trust_threshold_0.70"].sum())
        lines.append(f"Cells below the protocol's ~70% rolling-coverage trust threshold: "
                     f"**{n_below} / {len(coverage_df)}**. These are flagged, not dropped, "
                     f"per the task guardrails.")
    else:
        lines.append("(no coverage data available)")
    lines.append("")

    lines.append("## File locations")
    lines.append("")
    lines.append("- Raw snapshots: `data/raw/comtrade/<reporter>_<hsfamily>_<flow>_<period>.json`")
    lines.append("- Call manifest: `data/manifests/comtrade_manifest.jsonl`")
    lines.append("- Resume state: `data/manifests/comtrade_resume_state.json`")
    lines.append("- Long-format table: `data/interim/comtrade_trade_flows.parquet`")
    lines.append("- Coverage report: `data/interim/comtrade_coverage_report.csv`")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote status log -> {out_path}")
    return out_path


def main():
    summary, periods = run_ingestion()

    cfg = load_sources()
    ct_cfg = cfg["un_comtrade"]
    hs_families = ct_cfg["hs_families"]
    reporters = ct_cfg["reporters"]
    flows = ct_cfg["flows"]

    periods_attempted_by_combo = {}
    for reporter_iso in reporters:
        for hs_code in hs_families.values():
            for flow in flows:
                periods_attempted_by_combo[(reporter_iso, hs_code, flow)] = periods

    df, parquet_path = build_long_table()
    coverage_df, coverage_path = build_coverage_report(periods_attempted_by_combo)
    write_status_log(summary, coverage_df, periods)

    print("")
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    if summary["stop_reason"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

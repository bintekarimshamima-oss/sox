"""SEC EDGAR ingestion for the SOX fundamentals study.

Identifies the US semiconductor (SIC 3674) filer universe and pulls the
8 target us-gaap XBRL concepts (config/sources.yml: sec_edgar.concepts)
from each firm's companyfacts API, per config/protocol.yml's
release-aware ("filed" date) design.

Outputs
-------
data/manifests/sec_firm_universe.csv     firm universe + inclusion audit trail
data/manifests/sec_firm_universe_README.md   survivorship-bias note
data/raw/sec/<run_timestamp>/*.json      immutable raw companyfacts snapshots
data/manifests/sec_manifest.jsonl        provenance record per firm fetch
data/interim/sec_xbrl_facts.parquet      long-format fact table
data/interim/sec_concept_coverage_report.csv   per-concept coverage
outputs/logs/sec_ingest_status.md        run summary

Guardrails (config/protocol.yml, task instructions):
- never fabricate data on request failure; log and move on
- at most 3 retries per request, exponential backoff, then give up and log
- raw API responses are immutable; each run gets its own snapshot subdir
- do not truncate the firm universe to save time
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
import yaml
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "config" / "sources.yml"
PROTOCOL_PATH = ROOT / "config" / "protocol.yml"

MANIFEST_DIR = ROOT / "data" / "manifests"
INTERIM_DIR = ROOT / "data" / "interim"
RAW_SEC_DIR = ROOT / "data" / "raw" / "sec"
LOG_DIR = ROOT / "outputs" / "logs"

FIRM_UNIVERSE_CSV = MANIFEST_DIR / "sec_firm_universe.csv"
FIRM_UNIVERSE_README = MANIFEST_DIR / "sec_firm_universe_README.md"
SEC_MANIFEST_JSONL = MANIFEST_DIR / "sec_manifest.jsonl"
FACTS_PARQUET = INTERIM_DIR / "sec_xbrl_facts.parquet"
COVERAGE_CSV = INTERIM_DIR / "sec_concept_coverage_report.csv"
STATUS_MD = LOG_DIR / "sec_ingest_status.md"

ALLOWED_FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A"}
SLEEP_SECONDS = 0.15
BROWSE_PAGE_SIZE = 100  # SEC atom feed caps at 100 entries/page regardless of `count`
SAMPLE_WINDOW_START = pd.Period("2006Q1", freq="Q")
SAMPLE_WINDOW_END = pd.Period("2026Q2", freq="Q")
COVERAGE_FLAG_THRESHOLD = 0.70

RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_retryable_error(exc: BaseException) -> bool:
    """Retry on network errors and 5xx/429; do not retry on 404 (not-found is terminal)."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, (httpx.TransportError, httpx.TimeoutException))


@dataclass
class RunStats:
    universe_size: int = 0
    fetch_ok: int = 0
    fetch_failed: int = 0
    failures: list = field(default_factory=list)  # (cik, reason)
    total_fact_rows: int = 0
    concept_counts_per_firm: dict = field(default_factory=dict)


class SecClient:
    """Thin wrapper around httpx with SEC-compliant headers, rate limiting, and retries."""

    def __init__(self, user_agent: str):
        self.headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self.client = httpx.Client(timeout=30, headers=self.headers, follow_redirects=True)
        self._last_request_ts: Optional[float] = None

    def _throttle(self):
        if self._last_request_ts is not None:
            elapsed = time.monotonic() - self._last_request_ts
            if elapsed < SLEEP_SECONDS:
                time.sleep(SLEEP_SECONDS - elapsed)

    def get(self, url: str) -> httpx.Response:
        @retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            retry=retry_if_exception(is_retryable_error),
        )
        def _do_get():
            self._throttle()
            resp = self.client.get(url)
            self._last_request_ts = time.monotonic()
            # Raise for retry-eligible statuses so tenacity can back off; a plain 404
            # is returned as-is (caller decides -- it is an expected, terminal case).
            if resp.status_code == 429 or resp.status_code >= 500:
                resp.raise_for_status()
            return resp

        return _do_get()


# ---------------------------------------------------------------------------
# Step 1-2: firm universe via SIC 3674 browse-edgar atom feed + ticker mapping
# ---------------------------------------------------------------------------

ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}


def fetch_sic_cik_list(client: SecClient, sic: str) -> list[str]:
    """Paginate the SIC-3674 browse-edgar atom feed and return all distinct CIKs (zero-padded)."""
    ciks: list[str] = []
    seen = set()
    start = 0
    page = 1
    while True:
        url = (
            "https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&SIC={sic}&type=10-K&dateb=&owner=include"
            f"&count=200&start={start}&output=atom"
        )
        try:
            resp = client.get(url)
        except Exception as exc:
            print(f"[universe] FAILED page start={start}: {exc}", file=sys.stderr)
            break
        if resp.status_code != 200:
            print(f"[universe] page start={start} returned HTTP {resp.status_code}, stopping pagination")
            break

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as exc:
            print(f"[universe] XML parse error at start={start}: {exc}", file=sys.stderr)
            break

        entries = root.findall("a:entry", ATOM_NS)
        if not entries:
            break

        page_ciks = []
        for entry in entries:
            content = entry.find("a:content", ATOM_NS)
            if content is None:
                continue
            # The <company-info>/<cik> elements under <content> are emitted without
            # their own namespace prefix in the raw feed, but ElementTree still binds
            # them to the default Atom namespace since they inherit it from <feed>.
            company_info = content.find("a:company-info", ATOM_NS)
            if company_info is None:
                continue
            cik_el = company_info.find("a:cik", ATOM_NS)
            if cik_el is None or not cik_el.text:
                continue
            cik10 = cik_el.text.strip().zfill(10)
            if cik10 not in seen:
                seen.add(cik10)
                page_ciks.append(cik10)

        ciks.extend(page_ciks)
        print(f"[universe] page {page} (start={start}): {len(entries)} entries, "
              f"{len(page_ciks)} new CIKs (running total {len(ciks)})")

        if len(entries) < BROWSE_PAGE_SIZE:
            # Short page -> last page. (Some SEC deployments cap actual returned
            # entries at 100 even when count=200 is requested, so we detect the
            # end by a short page rather than assuming a fixed page size.)
            break

        start += len(entries)
        page += 1

    return ciks


def load_ticker_map(client: SecClient) -> dict:
    """Return {cik10: {"ticker": ..., "title": ...}} from SEC's bulk company_tickers.json."""
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = client.get(url)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        print(f"[universe] WARNING: could not fetch company_tickers.json: {exc}", file=sys.stderr)
        return {}
    mapping = {}
    for _, rec in raw.items():
        cik10 = str(rec["cik_str"]).zfill(10)
        mapping[cik10] = {"ticker": rec.get("ticker"), "title": rec.get("title")}
    return mapping


def build_firm_universe(client: SecClient, sic: str) -> pd.DataFrame:
    ciks = fetch_sic_cik_list(client, sic)
    ticker_map = load_ticker_map(client)

    rows = []
    for cik10 in ciks:
        tmap = ticker_map.get(cik10)
        if tmap:
            rows.append({
                "cik": cik10,
                "ticker": tmap["ticker"],
                "name": tmap["title"],
                "sic": sic,
                "source_note": "SIC 3674 browse-edgar atom feed; ticker/name via company_tickers.json",
            })
        else:
            rows.append({
                "cik": cik10,
                "ticker": None,
                "name": None,
                "sic": sic,
                "source_note": "SIC 3674 browse-edgar atom feed; not in company_tickers.json "
                                "(likely no active listed ticker); name backfilled from submissions API if available",
            })
    df = pd.DataFrame(rows).drop_duplicates(subset="cik").reset_index(drop=True)
    return df


def backfill_names_from_submissions(client: SecClient, universe: pd.DataFrame) -> pd.DataFrame:
    """For CIKs with no name from company_tickers.json, try the submissions API."""
    missing = universe[universe["name"].isna()]
    if missing.empty:
        return universe
    print(f"[universe] backfilling names for {len(missing)} CIKs with no ticker-map match via submissions API")
    universe = universe.copy()
    for idx, row in missing.iterrows():
        cik10 = row["cik"]
        url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
        try:
            resp = client.get(url)
        except Exception as exc:
            print(f"[universe] submissions fetch failed for CIK {cik10}: {exc}", file=sys.stderr)
            continue
        if resp.status_code == 404:
            universe.loc[idx, "source_note"] = row["source_note"] + "; submissions API 404 (possibly deregistered)"
            continue
        if resp.status_code != 200:
            universe.loc[idx, "source_note"] = row["source_note"] + f"; submissions API HTTP {resp.status_code}"
            continue
        try:
            data = resp.json()
        except ValueError:
            continue
        universe.loc[idx, "name"] = data.get("name")
        tickers = data.get("tickers") or []
        if tickers:
            universe.loc[idx, "ticker"] = tickers[0]
    return universe


# ---------------------------------------------------------------------------
# Step 3-5: companyfacts fetch + concept extraction + quality rules
# ---------------------------------------------------------------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_companyfacts(client: SecClient, cik10: str, raw_dir: Path) -> dict:
    """Fetch companyfacts JSON for one CIK. Returns a manifest-style dict; never raises."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    manifest_record = {
        "cik": cik10,
        "url": url,
        "retrieval_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": None,
        "http_status": None,
        "concept_count_found": 0,
    }
    try:
        resp = client.get(url)
    except Exception as exc:
        manifest_record["http_status"] = "ERROR"
        manifest_record["error"] = str(exc)
        return manifest_record, None

    manifest_record["http_status"] = resp.status_code

    if resp.status_code == 404:
        manifest_record["error"] = "404 not found (likely shell/inactive filer, no XBRL facts on file)"
        return manifest_record, None
    if resp.status_code != 200:
        manifest_record["error"] = f"unexpected HTTP {resp.status_code}"
        return manifest_record, None

    raw_bytes = resp.content
    manifest_record["sha256"] = sha256_bytes(raw_bytes)

    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"CIK{cik10}.json"
    if raw_path.exists():
        # Immutability guardrail: never overwrite a previously saved raw snapshot.
        raw_path = raw_dir / f"CIK{cik10}_{datetime.now(timezone.utc).strftime('%H%M%S%f')}.json"
    raw_path.write_bytes(raw_bytes)
    manifest_record["raw_path"] = str(raw_path.relative_to(ROOT))

    try:
        data = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        manifest_record["error"] = f"JSON decode error: {exc}"
        return manifest_record, None

    return manifest_record, data


def extract_concept_facts(cik10: str, companyfacts: dict, concepts: list[str]) -> list[dict]:
    """Pull the target us-gaap concepts' USD-unit facts, filtered to 10-Q/10-K forms."""
    rows = []
    gaap = (companyfacts.get("facts") or {}).get("us-gaap") or {}
    concepts_found = 0
    for concept in concepts:
        concept_data = gaap.get(concept)
        if not concept_data:
            continue
        concepts_found += 1
        units = concept_data.get("units") or {}
        for unit_name, facts in units.items():
            for fact in facts:
                form = fact.get("form")
                if form not in ALLOWED_FORMS:
                    continue
                rows.append({
                    "cik": cik10,
                    "concept": concept,
                    "unit": unit_name,
                    "val": fact.get("val"),
                    "start": fact.get("start"),
                    "end": fact.get("end"),
                    "fy": fact.get("fy"),
                    "fp": fact.get("fp"),
                    "form": form,
                    "filed": fact.get("filed"),
                    "frame": fact.get("frame"),
                    "accn": fact.get("accn"),
                    "is_amendment": "/A" in form,
                })
    return rows, concepts_found


def apply_dedup_rule(df: pd.DataFrame) -> pd.DataFrame:
    """When duplicate (cik, concept, end, fy, fp) rows exist from multiple accessions,
    keep the latest `filed` per accession -- but restatements are new information, so
    we keep every distinct `filed` date as its own row (do not backfill/overwrite).
    The only true duplicates collapsed here are identical (cik, concept, end, fy, fp,
    accn, val, filed) rows that can arise from a concept appearing in more than one
    unit block or being reported under more than one XBRL tag alias in the same filing.
    """
    if df.empty:
        return df
    before = len(df)
    dedup_cols = ["cik", "concept", "end", "fy", "fp", "accn", "val", "filed", "form"]
    df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"[dedup] dropped {before - after} exact-duplicate rows (same cik/concept/end/fy/fp/accn/val/filed)")
    return df


# ---------------------------------------------------------------------------
# Step 7: coverage report
# ---------------------------------------------------------------------------

def build_coverage_report(facts_df: pd.DataFrame, universe_df: pd.DataFrame, concepts: list[str]) -> pd.DataFrame:
    n_quarters_window = (SAMPLE_WINDOW_END - SAMPLE_WINDOW_START).n + 1
    n_firms = universe_df["cik"].nunique()
    total_firm_quarters_possible = n_firms * n_quarters_window

    rows = []
    for concept in concepts:
        sub = facts_df[(facts_df["concept"] == concept) & facts_df["val"].notna()]
        distinct_firms = sub["cik"].nunique()

        if sub.empty:
            firm_quarters = 0
        else:
            sub = sub.copy()
            end_dt = pd.to_datetime(sub["end"], errors="coerce")
            sub = sub.assign(_end_dt=end_dt)
            sub = sub.dropna(subset=["_end_dt"])
            sub["_quarter"] = sub["_end_dt"].dt.to_period("Q")
            in_window = sub[(sub["_quarter"] >= SAMPLE_WINDOW_START) & (sub["_quarter"] <= SAMPLE_WINDOW_END)]
            firm_quarters = in_window.drop_duplicates(subset=["cik", "_quarter"]).shape[0]

        coverage_ratio = firm_quarters / total_firm_quarters_possible if total_firm_quarters_possible else 0.0
        flagged = coverage_ratio < COVERAGE_FLAG_THRESHOLD

        rows.append({
            "concept": concept,
            "distinct_firms_with_data": distinct_firms,
            "firm_quarters_with_data": firm_quarters,
            "total_firm_quarters_possible": total_firm_quarters_possible,
            "coverage_ratio": round(coverage_ratio, 4),
            "below_70pct_threshold_FLAG": flagged,
            "note": ("LOW COVERAGE (<70%) -- flagged per protocol; exclusion decision "
                     "deferred to main analysis, not dropped here") if flagged else "",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# README / status writers
# ---------------------------------------------------------------------------

def write_firm_universe_readme(n_firms: int, n_with_ticker: int, sic: str):
    text = f"""# SEC firm universe -- survivorship-bias audit trail

`sec_firm_universe.csv` lists **every** CIK returned by the SEC EDGAR
`browse-edgar` full-text-search atom feed for SIC code {sic} (Semiconductors
& Related Devices) with at least one 10-K filing on record, as of the run
that produced this file. It is **not** limited to firms currently in the
PHLX Semiconductor (SOX) index.

## Why the universe is broader than the SOX index

Restricting the sample to *current* SOX constituents would introduce
survivorship bias: firms that were delisted, acquired, went bankrupt, or
were simply removed from the index over 2006-2026 would be silently
excluded, biasing fundamentals-based forecasts toward surviving firms.
Per the study protocol, the full SIC 3674 filer list is retained instead,
including:

- firms never in the SOX index (e.g. small-cap or foreign-private-issuer
  semiconductor filers under SIC 3674),
- firms delisted, merged, or acquired during the sample window,
- shell or inactive filers with an SIC 3674 registration but sparse/no
  XBRL data (these fail the companyfacts fetch and are logged, not
  fabricated).

Downstream index-membership filtering, if needed for a specific analysis,
should be applied explicitly and documented as its own step -- not baked
into this ingestion layer.

## Columns

- `cik` -- 10-digit zero-padded SEC Central Index Key
- `ticker` -- current ticker from SEC's `company_tickers.json` bulk file,
  or backfilled from the submissions API; null if the firm has no active
  listed ticker on record
- `name` -- registrant name
- `sic` -- SIC code (constant, "{sic}", by construction of the query)
- `source_note` -- how this row's identity fields were resolved, and any
  caveats (e.g. "not in company_tickers.json", "submissions API 404")

## Coverage note

Of {n_firms} distinct CIKs identified, {n_with_ticker} had a matching
entry in `company_tickers.json` (i.e. an active listed ticker at the time
of that bulk file's snapshot). The remainder are retained in the universe
with `ticker` possibly null -- this is expected for delisted/inactive/
foreign-private-issuer filers and is itself part of the audit trail, not
an error.
"""
    FIRM_UNIVERSE_README.write_text(text, encoding="utf-8")


def write_status_report(stats: RunStats, facts_df: pd.DataFrame, coverage_df: pd.DataFrame):
    lines = []
    lines.append("# SEC XBRL ingestion status\n")
    lines.append(f"- Run timestamp (UTC): {RUN_TIMESTAMP}")
    lines.append(f"- Firms in universe (SIC 3674, all 10-K filers on record): {stats.universe_size}")
    lines.append(f"- Firms successfully fetched (companyfacts HTTP 200 + parsed): {stats.fetch_ok}")
    lines.append(f"- Firms failed / skipped: {stats.fetch_failed}")
    lines.append(f"- Total fact-rows extracted (post quality filter + dedup): {stats.total_fact_rows}\n")

    if stats.failures:
        lines.append("## Failure reasons (CIK: reason)\n")
        for cik, reason in stats.failures:
            lines.append(f"- {cik}: {reason}")
        lines.append("")

    lines.append("## Concept coverage (sample window 2006 Q1 - 2026 Q2)\n")
    if not coverage_df.empty:
        header = "| " + " | ".join(coverage_df.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(coverage_df.columns)) + " |"
        lines.append(header)
        lines.append(sep)
        for _, r in coverage_df.iterrows():
            lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    else:
        lines.append("(no facts extracted)")
    lines.append("")

    flagged = coverage_df[coverage_df["below_70pct_threshold_FLAG"] == True] if not coverage_df.empty else pd.DataFrame()
    if not flagged.empty:
        lines.append("\n## Low-coverage flags (< 70% firm-quarter coverage)\n")
        lines.append("Per protocol, these are **flagged, not dropped**. The decision to "
                      "exclude any concept from the main analysis is made later, not by this ingestion script.\n")
        for _, row in flagged.iterrows():
            lines.append(f"- **{row['concept']}**: {row['coverage_ratio']:.1%} coverage")

    lines.append("\n## Output files\n")
    lines.append(f"- Firm universe: `data/manifests/sec_firm_universe.csv`")
    lines.append(f"- Firm universe README: `data/manifests/sec_firm_universe_README.md`")
    lines.append(f"- Raw snapshots: `data/raw/sec/{RUN_TIMESTAMP}/`")
    lines.append(f"- Fetch manifest: `data/manifests/sec_manifest.jsonl`")
    lines.append(f"- Facts table: `data/interim/sec_xbrl_facts.parquet`")
    lines.append(f"- Coverage report: `data/interim/sec_concept_coverage_report.csv`")

    STATUS_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_yaml(CONFIG_PATH)
    sec_cfg = cfg["sec_edgar"]
    user_agent = sec_cfg["user_agent"]
    sic = sec_cfg["sic_target"]
    concepts = sec_cfg["concepts"]

    for d in (MANIFEST_DIR, INTERIM_DIR, RAW_SEC_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    run_raw_dir = RAW_SEC_DIR / RUN_TIMESTAMP
    run_raw_dir.mkdir(parents=True, exist_ok=True)

    client = SecClient(user_agent)
    stats = RunStats()

    # --- Step 1-2: firm universe ---
    print(f"=== Step 1-2: building SIC {sic} firm universe ===")
    universe_df = build_firm_universe(client, sic)
    universe_df = backfill_names_from_submissions(client, universe_df)
    stats.universe_size = len(universe_df)
    n_with_ticker = universe_df["ticker"].notna().sum()

    universe_df.to_csv(FIRM_UNIVERSE_CSV, index=False)
    write_firm_universe_readme(stats.universe_size, int(n_with_ticker), sic)
    print(f"[universe] wrote {stats.universe_size} firms -> {FIRM_UNIVERSE_CSV}")

    # --- Step 3-5: companyfacts fetch + extraction ---
    print(f"\n=== Step 3-5: fetching companyfacts for {stats.universe_size} firms ===")
    all_rows = []
    manifest_records = []
    for i, row in enumerate(universe_df.itertuples(index=False), start=1):
        cik10 = row.cik
        manifest_record, data = fetch_companyfacts(client, cik10, run_raw_dir)

        if data is None:
            stats.fetch_failed += 1
            reason = manifest_record.get("error", f"HTTP {manifest_record.get('http_status')}")
            stats.failures.append((cik10, reason))
            manifest_records.append(manifest_record)
            if i % 25 == 0 or i == stats.universe_size:
                print(f"  [{i}/{stats.universe_size}] CIK {cik10}: FAILED ({reason})")
            continue

        firm_rows, concepts_found = extract_concept_facts(cik10, data, concepts)
        manifest_record["concept_count_found"] = concepts_found
        manifest_records.append(manifest_record)

        if firm_rows:
            all_rows.extend(firm_rows)
            stats.fetch_ok += 1
        else:
            # Fetched fine, but none of the 8 target concepts present (e.g. non-10-Q/K
            # forms only, or a firm that doesn't report these particular tags).
            stats.fetch_ok += 1

        if i % 25 == 0 or i == stats.universe_size:
            print(f"  [{i}/{stats.universe_size}] CIK {cik10}: {concepts_found}/{len(concepts)} "
                  f"concepts found, {len(firm_rows)} rows extracted")

    with open(SEC_MANIFEST_JSONL, "a", encoding="utf-8") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\n[manifest] appended {len(manifest_records)} records -> {SEC_MANIFEST_JSONL}")

    # --- Step 6: quality rules + write facts parquet ---
    print("\n=== Step 6: quality filtering + writing facts table ===")
    facts_df = pd.DataFrame(all_rows)
    if not facts_df.empty:
        assert facts_df["form"].isin(ALLOWED_FORMS).all(), "form filter leaked disallowed forms"
        facts_df = apply_dedup_rule(facts_df)
    stats.total_fact_rows = len(facts_df)

    if not facts_df.empty:
        facts_df.to_parquet(FACTS_PARQUET, engine="pyarrow", index=False)
    else:
        # Still write an empty, correctly-typed parquet so downstream steps don't crash.
        empty_cols = ["cik", "concept", "unit", "val", "start", "end", "fy", "fp",
                      "form", "filed", "frame", "accn", "is_amendment"]
        pd.DataFrame(columns=empty_cols).to_parquet(FACTS_PARQUET, engine="pyarrow", index=False)
    print(f"[facts] wrote {stats.total_fact_rows} rows -> {FACTS_PARQUET}")

    # --- Step 7: coverage report ---
    print("\n=== Step 7: concept coverage report ===")
    coverage_df = build_coverage_report(facts_df, universe_df, concepts)
    coverage_df.to_csv(COVERAGE_CSV, index=False)
    print(f"[coverage] wrote -> {COVERAGE_CSV}")
    print(coverage_df.to_string(index=False))

    # --- status log ---
    write_status_report(stats, facts_df, coverage_df)
    print(f"\n[status] wrote -> {STATUS_MD}")

    print("\n=== DONE ===")
    print(f"Universe: {stats.universe_size} | OK: {stats.fetch_ok} | Failed: {stats.fetch_failed} | "
          f"Fact rows: {stats.total_fact_rows}")


if __name__ == "__main__":
    main()

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw" / "fred"
MANIFEST_PATH = ROOT / "data" / "manifests" / "fred_manifest.jsonl"


def load_sources():
    with open(ROOT / "config" / "sources.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_series(series_id: str) -> str:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    text = resp.text
    if not text.startswith("observation_date") and not text.startswith("DATE"):
        raise ValueError(f"Unexpected FRED response for {series_id}: {text[:200]}")
    return text, url


def main():
    cfg = load_sources()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for series_id, meta in cfg["fred"]["series"].items():
        text, url = fetch_series(series_id)
        out_path = RAW_DIR / f"{series_id}.csv"
        out_path.write_text(text, encoding="utf-8")
        sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row_count = text.count("\n")
        record = {
            "source": "fred",
            "series_id": series_id,
            "role": meta.get("role"),
            "url": url,
            "retrieval_utc": datetime.now(timezone.utc).isoformat(),
            "licence_note": "FRED public data, St. Louis Fed terms of use",
            "sha256": sha256,
            "row_count": row_count,
            "raw_path": str(out_path.relative_to(ROOT)),
        }
        records.append(record)
        print(f"Fetched {series_id}: {row_count} rows -> {out_path.name}")

    with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote manifest entries for {len(records)} FRED series.")


if __name__ == "__main__":
    main()

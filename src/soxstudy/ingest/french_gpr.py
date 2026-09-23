import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = ROOT / "data" / "manifests" / "french_gpr_manifest.jsonl"
HEADERS = {"User-Agent": "Mozilla/5.0 PhD Research yusha@iscb.org"}


def manifest_record(source, url, path, content_bytes, note):
    return {
        "source": source,
        "url": url,
        "retrieval_utc": datetime.now(timezone.utc).isoformat(),
        "licence_note": note,
        "sha256": hashlib.sha256(content_bytes).hexdigest(),
        "byte_count": len(content_bytes),
        "raw_path": str(path.relative_to(ROOT)),
    }


def fetch_french():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    resp = httpx.get(url, timeout=30, follow_redirects=True, headers=HEADERS)
    resp.raise_for_status()
    out_dir = RAW_DIR / "french"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "F-F_Research_Data_5_Factors_2x3_CSV.zip"
    zip_path.write_bytes(resp.content)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        names = z.namelist()
        for name in names:
            z.extract(name, out_dir)
    print("French factors extracted:", names)
    return manifest_record(
        "kenneth_french_5_factors", url, zip_path, resp.content,
        "Kenneth R. French Data Library, academic public use"
    )


def fetch_gpr():
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
    resp = httpx.get(url, timeout=30, follow_redirects=True, headers=HEADERS)
    resp.raise_for_status()
    out_dir = RAW_DIR / "gpr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_gpr_export.xls"
    out_path.write_bytes(resp.content)
    print("GPR file saved:", len(resp.content), "bytes")
    return manifest_record(
        "caldara_iacoviello_gpr", url, out_path, resp.content,
        "Caldara & Iacoviello GPR dataset, public academic dataset"
    )


def main():
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = [fetch_french(), fetch_gpr()]
    with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} manifest entries.")


if __name__ == "__main__":
    main()

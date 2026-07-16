from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from .config import (
    NHANES_FILES,
    OFFICIAL_FALLBACK_BASE_URL,
    RAW_DATA_DIR,
    REQUESTED_BASE_URL,
)

LOGGER = logging.getLogger(__name__)
XPT_SIGNATURE = b"HEADER RECORD"


def _looks_like_xpt(content: bytes) -> bool:
    return content.lstrip().startswith(XPT_SIGNATURE)


def _download_one(url: str, destination: Path, timeout: int = 120) -> tuple[bool, str]:
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with requests.get(url, stream=True, timeout=(30, max(timeout, 300))) as response:
            response.raise_for_status()
            chunks = response.iter_content(chunk_size=256 * 1024)
            first_chunk = next((chunk for chunk in chunks if chunk), b"")
            if not first_chunk:
                return False, "Empty response"
            if not _looks_like_xpt(first_chunk):
                content_type = response.headers.get("content-type", "unknown")
                return (
                    False,
                    f"Response was not an XPT file (content-type={content_type})",
                )
            bytes_written = 0
            with temporary.open("wb") as handle:
                handle.write(first_chunk)
                bytes_written += len(first_chunk)
                for chunk in chunks:
                    if not chunk:
                        continue
                    handle.write(chunk)
                    bytes_written += len(chunk)
    except requests.RequestException as exc:
        temporary.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: {exc}"
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: {exc}"

    temporary.replace(destination)
    return True, f"downloaded {bytes_written:,} bytes"


def download_nhanes_files(
    raw_dir: Path = RAW_DATA_DIR,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Download official NHANES files, validating the SAS XPORT signature."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for component, filename in tqdm(NHANES_FILES.items(), desc="NHANES files"):
        destination = raw_dir / filename
        if destination.exists() and not overwrite:
            if _looks_like_xpt(destination.read_bytes()[:128]):
                records.append(
                    {
                        "component": component,
                        "filename": filename,
                        "status": "available",
                        "source_url": f"{OFFICIAL_FALLBACK_BASE_URL}/{filename}",
                        "reason": "valid cached XPT",
                    }
                )
                continue
            destination.unlink()

        attempts = [
            f"{REQUESTED_BASE_URL}/{filename}",
            f"{OFFICIAL_FALLBACK_BASE_URL}/{filename}",
        ]
        reasons: list[str] = []
        success = False
        source_url = ""
        for url in attempts:
            ok, reason = _download_one(url, destination)
            reasons.append(f"{url}: {reason}")
            if ok:
                success = True
                source_url = url
                LOGGER.info("Downloaded %s from %s", filename, url)
                break
            LOGGER.warning("Download attempt failed for %s: %s", filename, reason)

        records.append(
            {
                "component": component,
                "filename": filename,
                "status": "downloaded" if success else "missing",
                "source_url": source_url,
                "reason": " | ".join(reasons),
            }
        )

    return pd.DataFrame(records)

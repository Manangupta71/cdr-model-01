"""
streetview_probe.py

Probes Google Street View coverage and extracts visual environmental indices
(greenery via Excess Green index and edge density) for inferred home anchors.

Security & Integrity:
  - Google Maps API key is read EXCLUSIVELY from the GOOGLE_MAPS_API_KEY
    environment variable. Never hardcode or log the API key.
  - Image fetching and index computation are executed strictly IN-MEMORY.
    No image files are saved to disk.
  - Uses exclusively inferred home anchors from output/home_work_anchors.csv.
    Never accesses ground-truth coordinates from population.csv.

Endpoints:
  - Step 1 (Default): Street View Metadata API (free endpoint) to check coverage,
    pano_id, and capture date.
  - Step 2 (--fetch): Street View Static API to fetch 4 cardinal headings
    (0, 90, 180, 270 degrees) and compute Excess Green (ExG) and edge density.
"""

import os
import io
import time
import argparse
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageFilter

from spatial_context import _resolve_phone_numbers

METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
STATIC_URL = "https://maps.googleapis.com/maps/api/streetview"
HEADINGS = [0, 90, 180, 270]


def compute_excess_green_index(img: Image.Image) -> float:
    """
    Computes normalized Excess Green Index (ExG = 2g - r - b) across RGB channels.
    Higher values indicate denser canopy / vegetation.
    """
    arr = np.asarray(img.convert("RGB"), dtype=float)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    total = r + g + b + 1e-6
    rn = r / total
    gn = g / total
    bn = b / total
    exg = 2.0 * gn - rn - bn
    return float(np.mean(exg))


def compute_edge_density(img: Image.Image, threshold: float = 0.15) -> float:
    """
    Computes edge density (proportion of high-gradient edge pixels) in-memory
    using a Laplacian/Sobel edge filter. Stand-in for visual structural complexity.
    """
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.asarray(edges, dtype=float) / 255.0
    return float(np.mean(edge_arr > threshold))


def probe_metadata(
    lat: float,
    lon: float,
    api_key: Optional[str],
    dry_run: bool = False,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Queries Street View Metadata API for coverage and pano info.
    """
    if dry_run:
        # Mock metadata for dry run without making network calls
        return {
            "status": "OK",
            "has_coverage": 1,
            "pano_id": f"dryrun_pano_{abs(hash((lat, lon))) % 1000000:06d}",
            "date": "2025-06",
        }

    params = {"location": f"{lat:.6f},{lon:.6f}", "key": api_key}
    resp = requests.get(METADATA_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    status = data.get("status", "ZERO_RESULTS")
    has_cov = 1 if status == "OK" else 0
    return {
        "status": status,
        "has_coverage": has_cov,
        "pano_id": data.get("pano_id", ""),
        "date": data.get("date", ""),
    }


def fetch_and_compute_indices(
    lat: float,
    lon: float,
    api_key: Optional[str],
    dry_run: bool = False,
    rate_limit: float = 0.1,
    timeout: float = 10.0,
) -> Tuple[float, float, float, float, int]:
    """
    Fetches 4 cardinal headings in-memory and computes greenery (ExG) and edge density.
    Returns: (exg_mean, exg_std, edge_mean, edge_std, requests_made).
    """
    exg_vals = []
    edge_vals = []
    requests_made = 0

    for heading in HEADINGS:
        if dry_run:
            # Deterministic synthetic in-memory image for dry run
            rng = np.random.default_rng(abs(hash((lat, lon, heading))) % (2**31))
            mock_pixels = (rng.uniform(0, 255, (100, 100, 3))).astype(np.uint8)
            img = Image.fromarray(mock_pixels)
        else:
            params = {
                "size": "400x400",
                "location": f"{lat:.6f},{lon:.6f}",
                "heading": heading,
                "pitch": 0,
                "key": api_key,
            }
            resp = requests.get(STATIC_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
            requests_made += 1
            if rate_limit > 0:
                time.sleep(rate_limit)

        exg_vals.append(compute_excess_green_index(img))
        edge_vals.append(compute_edge_density(img))

    return (
        float(np.mean(exg_vals)),
        float(np.std(exg_vals)),
        float(np.mean(edge_vals)),
        float(np.std(edge_vals)),
        requests_made,
    )


def run_probe(
    anchors_path: str = "output/home_work_anchors.csv",
    output_path: str = "output/streetview_features.csv",
    fetch_images: bool = False,
    max_requests: int = 500,
    rate_limit: float = 0.1,
    dry_run: bool = False,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Runs metadata probe and optional image feature computation.
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not dry_run and not api_key:
        raise ValueError(
            "GOOGLE_MAPS_API_KEY environment variable is not set. "
            "Set it in your shell environment or pass --dry_run for testing."
        )

    if not os.path.exists(anchors_path):
        raise FileNotFoundError(f"Inferred anchors file not found: {anchors_path}")

    anchors = pd.read_csv(anchors_path)
    anchors = _resolve_phone_numbers(anchors)

    existing_df = None
    processed_phones = set()
    if resume and os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            if "phone_number" in existing_df.columns:
                processed_phones = set(existing_df["phone_number"].dropna().tolist())
                print(f"Resuming from {output_path} ({len(processed_phones)} records already processed).")
        except Exception:
            existing_df = None

    records: List[Dict[str, Any]] = []
    total_requests = 0

    print(f"Starting Street View probe (dry_run={dry_run}, fetch_images={fetch_images}, "
          f"max_requests={max_requests}, rate_limit={rate_limit}s)...")

    for row in anchors.itertuples():
        phone = row.phone_number
        if phone in processed_phones:
            continue

        if total_requests >= max_requests:
            print(f"Reached hard request cap ({max_requests} requests). Stopping probe early.")
            break

        lat = float(row.inferred_home_lat)
        lon = float(row.inferred_home_lon)

        # Step 1: Metadata probe
        meta = probe_metadata(lat, lon, api_key, dry_run=dry_run)
        total_requests += 1 if not dry_run else 0
        if rate_limit > 0 and not dry_run:
            time.sleep(rate_limit)

        rec = {
            "phone_number": phone,
            "sv_status": meta["status"],
            "sv_has_coverage": meta["has_coverage"],
            "sv_pano_id": meta["pano_id"],
            "sv_date": meta["date"],
            "sv_exg_mean": np.nan,
            "sv_exg_std": np.nan,
            "sv_edge_density_mean": np.nan,
            "sv_edge_density_std": np.nan,
        }

        # Step 2: In-memory image fetch and feature extraction
        if fetch_images and meta["has_coverage"] == 1:
            if total_requests + 4 > max_requests:
                print(f"Request cap approaching ({total_requests}/{max_requests}). Skipping image fetch for {phone}.")
            else:
                exg_m, exg_s, edge_m, edge_s, reqs = fetch_and_compute_indices(
                    lat, lon, api_key, dry_run=dry_run, rate_limit=rate_limit
                )
                total_requests += reqs
                rec["sv_exg_mean"] = round(exg_m, 5)
                rec["sv_exg_std"] = round(exg_s, 5)
                rec["sv_edge_density_mean"] = round(edge_m, 5)
                rec["sv_edge_density_std"] = round(edge_s, 5)

        records.append(rec)

    new_df = pd.DataFrame(records)
    if existing_df is not None and not existing_df.empty:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Saved {len(final_df)} records to {output_path} (total requests made: {total_requests}).")
    return final_df


def main():
    parser = argparse.ArgumentParser(description="Google Street View metadata and image feature probe.")
    parser.add_argument("--anchors", type=str, default="output/home_work_anchors.csv",
                        help="Path to home_work_anchors.csv.")
    parser.add_argument("--out", type=str, default="output/streetview_features.csv",
                        help="Output path for streetview_features.csv.")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch 4 headings per covered point in-memory and compute ExG and edge density.")
    parser.add_argument("--max_requests", type=int, default=500,
                        help="Hard cap on total API requests.")
    parser.add_argument("--rate_limit", type=float, default=0.1,
                        help="Delay in seconds between requests for polite rate limiting.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Simulate API interactions in-memory without making network requests.")
    parser.add_argument("--no_resume", action="store_true",
                        help="Do not resume from existing partial output file.")
    args = parser.parse_args()

    run_probe(
        anchors_path=args.anchors,
        output_path=args.out,
        fetch_images=args.fetch,
        max_requests=args.max_requests,
        rate_limit=args.rate_limit,
        dry_run=args.dry_run,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

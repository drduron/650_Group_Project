"""Download the Sparkov credit-card fraud dataset from Kaggle into data/.

Usage:
    python download_data.py                          # -> data/fraudTrain.csv and data/fraudTest.csv (~500 MB)
    python download_data.py --zip fraud-detection.zip  # use a zip you downloaded yourself

Source: https://www.kaggle.com/datasets/kartik2112/fraud-detection (licence: CC0 Public Domain).
The dataset is public, and Kaggle's download endpoint normally serves it without an account. If it asks you to
sign in, download the zip from the page above (or run `kaggle datasets download -d kartik2112/fraud-detection`
with the Kaggle CLI) and pass it with --zip.

Uses only the Python standard library. Only the two expected CSV files are extracted, and their row counts are
checked against the published dataset.
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

import config

URL = 'https://www.kaggle.com/api/v1/datasets/download/kartik2112/fraud-detection'
EXPECTED_ROWS = {config.TRAIN_FILE: 1_296_675, config.TEST_FILE: 555_719}   # data rows, excluding the header


def count_rows(path):
    """Number of lines minus the header (the files contain no quoted line breaks)."""
    with open(path, 'rb') as f:
        return sum(chunk.count(b'\n') for chunk in iter(lambda: f.read(1 << 24), b'')) - 1


def is_complete(data_dir):
    return all((data_dir / name).exists() and count_rows(data_dir / name) == n for name, n in EXPECTED_ROWS.items())


def download(url, dest):
    """Stream `url` to `dest`, printing progress."""
    request = urllib.request.Request(url, headers={'User-Agent': 'fraudwatch-download/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response, open(dest, 'wb') as out:
        total = int(response.headers.get('Content-Length') or 0)
        done, next_report = 0, 0.1
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total and done / total >= next_report:
                print(f'  {done / total:4.0%} of {total / 1e6:.0f} MB', flush=True)
                next_report += 0.1


def extract(zip_path, data_dir):
    """Extract exactly the expected CSVs (nothing else, no paths taken from the archive)."""
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        missing = [n for n in EXPECTED_ROWS if n not in names]
        if missing:
            sys.exit(f'{zip_path} does not contain {missing}; found {sorted(names)}')
        bad = z.testzip()
        if bad:
            sys.exit(f'{zip_path} is corrupt (first bad file: {bad})')
        for name in EXPECTED_ROWS:
            with z.open(name) as src, open(data_dir / name, 'wb') as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', type=Path, default=config.DATA_DIR, help='where to put the CSVs (default: data/)')
    ap.add_argument('--zip', type=Path, help='use an already-downloaded zip instead of downloading')
    ap.add_argument('--force', action='store_true', help='download again even if the files are already there')
    args = ap.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    if is_complete(args.data_dir) and not args.force:
        print(f'data already present and complete in {config.display_path(args.data_dir)}')
        return
    zip_path = args.zip
    if zip_path is None:
        zip_path = args.data_dir / 'fraud-detection.zip'
        print(f'downloading {URL}')
        try:
            download(URL, zip_path)
        except Exception as e:  # network errors, HTTP 401/403, ...
            zip_path.unlink(missing_ok=True)
            sys.exit(f'download failed ({e}). Download the zip manually from '
                     f'https://www.kaggle.com/datasets/kartik2112/fraud-detection and run: python download_data.py --zip <file>')
    print(f'extracting {list(EXPECTED_ROWS)} to {config.display_path(args.data_dir)}')
    extract(zip_path, args.data_dir)
    if args.zip is None:
        zip_path.unlink()
    for name, n in EXPECTED_ROWS.items():
        rows = count_rows(args.data_dir / name)
        print(f'  {name}: {rows:,} rows {"(ok)" if rows == n else f"(EXPECTED {n:,})"}')
    if not is_complete(args.data_dir):
        sys.exit('row counts do not match the published dataset')
    print('done')


if __name__ == '__main__':
    main()

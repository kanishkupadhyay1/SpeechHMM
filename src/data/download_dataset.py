"""
Dataset download module for Google Speech Commands Dataset v0.02.
Official source: https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
"""

import os
import sys
import tarfile
import hashlib
import time
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATASET_URL = "https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
EXPECTED_MD5 = "a3TzkBIUyywpNOmBloKYNQ=="  # Base64 MD5 from Google GCS header: a3TzkBIUyywpNOmBloKYNQ==
# In hex: a3TzkBIUyywpNOmBloKYNQ== -> 6b74f3901214cb2c2934e98196829835
EXPECTED_MD5_HEX = "6b74f3901214cb2c2934e98196829835"
EXPECTED_SIZE = 2428923189  # bytes (~2.26 GB)

# 10 core commands specified in requirements:
CORE_COMMANDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
]

# Additional classes for unknown and silence modeling:
AUXILIARY_CLASSES = [
    "_background_noise_", "zero", "one", "two", "three", "four"
]


def download_file(url: str, dest_path: str, chunk_size: int = 1024 * 1024) -> None:
    """Download a file with resume support and progress logging."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".part"
    downloaded_bytes = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0

    headers = {}
    if downloaded_bytes > 0:
        headers["Range"] = f"bytes={downloaded_bytes}-"
        print(f"Resuming download from byte {downloaded_bytes} ({downloaded_bytes / (1024*1024):.1f} MB)...")
    else:
        print(f"Starting download from {url}...")

    response = requests.get(url, stream=True, headers=headers, verify=False, timeout=60)
    
    if response.status_code == 416:
        # Range not satisfiable, file might be complete
        print("Range not satisfiable. Checking file...")
    elif response.status_code not in (200, 206):
        raise RuntimeError(f"Failed to download: HTTP status code {response.status_code}")

    total_size = None
    if "content-range" in response.headers:
        total_size = int(response.headers["content-range"].split("/")[-1])
    elif "content-length" in response.headers:
        total_size = int(response.headers["content-length"]) + downloaded_bytes

    mode = "ab" if downloaded_bytes > 0 else "wb"
    start_time = time.time()
    last_print_time = start_time
    written_bytes = downloaded_bytes

    with open(temp_path, mode) as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                written_bytes += len(chunk)
                curr_time = time.time()
                if curr_time - last_print_time >= 5.0 or (total_size and written_bytes >= total_size):
                    elapsed = curr_time - start_time
                    speed = (written_bytes - downloaded_bytes) / (elapsed + 1e-6) / (1024 * 1024)
                    if total_size:
                        pct = (written_bytes / total_size) * 100
                        print(f"Progress: {written_bytes / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB "
                              f"({pct:.1f}%) - Speed: {speed:.2f} MB/s")
                    else:
                        print(f"Downloaded: {written_bytes / (1024*1024):.1f} MB - Speed: {speed:.2f} MB/s")
                    last_print_time = curr_time

    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(temp_path, dest_path)
    print(f"Download complete: {dest_path} ({os.path.getsize(dest_path) / (1024*1024):.1f} MB)")


def verify_md5(file_path: str, expected_hex: str) -> bool:
    """Verify md5 checksum of file."""
    print("Verifying MD5 checksum of downloaded archive...")
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            md5.update(chunk)
    digest = md5.hexdigest()
    print(f"Calculated MD5: {digest}")
    print(f"Expected MD5:   {expected_hex}")
    return digest.lower() == expected_hex.lower()


def extract_dataset(tar_path: str, extract_to: str, filter_classes: bool = False) -> None:
    """
    Extract tar.gz archive.
    If filter_classes is True, extracts the 10 core commands, background noise,
    and split text files to optimize extraction speed and disk usage.
    """
    os.makedirs(extract_to, exist_ok=True)
    print(f"Extracting archive {tar_path} into {extract_to}...")

    allowed_prefixes = tuple(CORE_COMMANDS + AUXILIARY_CLASSES + ["_background_noise_"])
    allowed_files = {"testing_list.txt", "validation_list.txt", "LICENSE", "README.md"}

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        print(f"Total members in archive: {len(members)}")

        selected_members = []
        for m in members:
            # Check if member belongs to core commands or auxiliary files
            name = m.name.lstrip("./\\").replace("\\", "/")
            parts = name.split("/")
            first_dir = parts[0]

            if not filter_classes:
                selected_members.append(m)
            else:
                if len(parts) == 1 and parts[0] in allowed_files:
                    selected_members.append(m)
                elif first_dir in allowed_prefixes:
                    selected_members.append(m)

        print(f"Extracting {len(selected_members)} files...")
        tar.extractall(path=extract_to, members=selected_members)
    print("Extraction complete.")


def run_download_and_extract(raw_dir: str, filter_classes: bool = False) -> None:
    """Orchestrate download and extraction."""
    archive_path = os.path.join(raw_dir, "speech_commands_v0.02.tar.gz")
    dataset_dir = os.path.join(raw_dir, "speech_commands_v0.02")

    # Check if already extracted
    val_file = os.path.join(dataset_dir, "validation_list.txt")
    test_file = os.path.join(dataset_dir, "testing_list.txt")
    if os.path.exists(val_file) and os.path.exists(test_file):
        print(f"Dataset already extracted in {dataset_dir}")
        return

    # Check if archive exists
    if not os.path.exists(archive_path) or os.path.getsize(archive_path) < 1000000:
        download_file(DATASET_URL, archive_path)
    else:
        print(f"Archive already found at {archive_path} ({os.path.getsize(archive_path)/(1024*1024):.1f} MB)")

    extract_dataset(archive_path, dataset_dir, filter_classes=filter_classes)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download and extract Google Speech Commands v0.02")
    parser.add_argument("--raw-dir", default=os.path.join("data", "raw"), help="Target directory for raw data")
    parser.add_argument("--all-classes", action="store_true", help="Extract all 35 classes instead of core 10 + aux")
    args = parser.parse_args()

    run_download_and_extract(args.raw_dir, filter_classes=not args.all_classes)

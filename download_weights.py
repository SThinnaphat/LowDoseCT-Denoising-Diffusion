"""
download_weights.py
-------------------
Downloads pre-trained FA-CFG-LDCT checkpoints from Google Drive.

Usage:
    python download_weights.py

Downloads all checkpoints into ./checkpoints/
"""

import os
import urllib.request

# ── Checkpoint metadata ────────────────────────────────────────────────────
# Replace each GDRIVE_FILE_ID with the actual file ID from your Google Drive
# share link: https://drive.google.com/file/d/FILE_ID/view
CHECKPOINTS = {
    "config_A_best.pt": {
        "id":   "GDRIVE_FILE_ID_CONFIG_A",
        "desc": "Baseline DDPM (MSE only, no CFG, no FANP)"
    },
    "config_B_best.pt": {
        "id":   "GDRIVE_FILE_ID_CONFIG_B",
        "desc": "Config B: +CFG only (w=1.5 at inference)"
    },
    "config_C_best.pt": {
        "id":   "GDRIVE_FILE_ID_CONFIG_C",
        "desc": "Config C: +FANP only (lambda=0.01)"
    },
    "config_D_best.pt": {
        "id":   "GDRIVE_FILE_ID_CONFIG_D",
        "desc": "Config D: Proposed (CFG + FANP combined)"
    },
    "p2p_best.pt": {
        "id":   "GDRIVE_FILE_ID_P2P",
        "desc": "Pix2Pix GAN baseline"
    },
    "wgan_best.pt": {
        "id":   "GDRIVE_FILE_ID_WGAN",
        "desc": "WGAN-VGG GAN baseline"
    },
}

SAVE_DIR = "./checkpoints"


def gdrive_url(file_id):
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_file(url, dest_path, desc=""):
    print(f"  Downloading {os.path.basename(dest_path)} — {desc}")
    try:
        urllib.request.urlretrieve(url, dest_path)
        size_mb = os.path.getsize(dest_path) / 1e6
        print(f"  Saved to {dest_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  Failed: {e}")
        print(f"  Please download manually from: {url}")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Downloading checkpoints to {SAVE_DIR}/\n")

    for filename, meta in CHECKPOINTS.items():
        dest = os.path.join(SAVE_DIR, filename)

        if os.path.exists(dest):
            print(f"  {filename} already exists, skipping.")
            continue

        if "GDRIVE_FILE_ID" in meta["id"]:
            print(f"  SKIP {filename} — Google Drive ID not set.")
            print(f"         Edit download_weights.py and replace "
                  f"GDRIVE_FILE_ID_{filename.split('_best')[0].upper()}")
            print(f"         with the file ID from your Google Drive share link.\n")
            continue

        url = gdrive_url(meta["id"])
        download_file(url, dest, meta["desc"])

    print("\nDone.")
    print("\nTo set up the Google Drive IDs:")
    print("  1. Upload each .pt file to Google Drive")
    print("  2. Right-click the file > Share > Copy link")
    print("  3. Extract the file ID from the link:")
    print("     https://drive.google.com/file/d/FILE_ID_HERE/view")
    print("  4. Replace GDRIVE_FILE_ID_XXX in this script with that ID")


if __name__ == "__main__":
    main()

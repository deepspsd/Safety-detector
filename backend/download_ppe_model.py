#!/usr/bin/env python3
"""
download_ppe_model.py
=====================
Downloads the custom ppe.pt model from Google Drive.

Source: https://github.com/biswadeep-roy/Safety-Detection-YOLOv8
Model:  ppe.pt  — YOLOv8 trained on PPE dataset (10 classes)

Classes detected by ppe.pt:
  0: Hardhat        ✅ compliant
  1: Mask           ✅ compliant
  2: NO-Hardhat     ❌ violation
  3: NO-Mask        ❌ violation
  4: NO-Safety Vest ❌ violation
  5: Person         👤 person
  6: Safety Cone    🟠 neutral
  7: Safety Vest    ✅ compliant
  8: machinery      🔵 neutral
  9: vehicle        🔵 neutral

Usage:
  python download_ppe_model.py

After running, ppe.pt will be placed in:
  backend/ppe.pt   ← the detection service loads it from here
"""

import os
import sys

# ── Google Drive folder ID from the repo README ──────────────────
# https://drive.google.com/drive/folders/11tfTBkp4JdlJ8QXoAMZxgVMf8xpLBXi_
GDRIVE_FOLDER_ID = "11tfTBkp4JdlJ8QXoAMZxgVMf8xpLBXi_"
OUTPUT_FILE = "ppe.pt"

def download_with_gdown():
    try:
        import gdown
    except ImportError:
        print("Installing gdown…")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown

    print(f"📥 Downloading ppe.pt from Google Drive…")
    print(f"   Folder ID: {GDRIVE_FOLDER_ID}")

    # Try direct folder download — picks ppe.pt automatically
    url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
    try:
        gdown.download_folder(url, quiet=False, use_cookies=False, output="./")
        # Move .pt file if it landed in a subfolder
        for root, dirs, files in os.walk("."):
            for f in files:
                if f.endswith(".pt") and f != OUTPUT_FILE:
                    src = os.path.join(root, f)
                    os.rename(src, OUTPUT_FILE)
                    print(f"✅ Model saved as: {OUTPUT_FILE}")
                    return True
        if os.path.exists(OUTPUT_FILE):
            print(f"✅ ppe.pt downloaded successfully!")
            sz = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
            print(f"   Size: {sz:.1f} MB")
            return True
    except Exception as e:
        print(f"   Folder download failed: {e}")

    return False


def manual_instructions():
    print("\n" + "─" * 60)
    print("📋 MANUAL DOWNLOAD INSTRUCTIONS")
    print("─" * 60)
    print("1. Open this URL in your browser:")
    print("   https://drive.google.com/drive/folders/11tfTBkp4JdlJ8QXoAMZxgVMf8xpLBXi_")
    print()
    print("2. Download  ppe.pt  from that folder")
    print()
    print("3. Place it in your backend directory:")
    print("   DETECTION MODEL/backend/ppe.pt")
    print()
    print("4. Restart the backend (uvicorn main:app --reload --port 8000)")
    print("   You should see:  ✅ Custom ppe.pt model loaded (PPE-native, 10 classes)")
    print("─" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("  Safety Monitor — ppe.pt Model Downloader")
    print("  Source: biswadeep-roy/Safety-Detection-YOLOv8")
    print("=" * 60)

    if os.path.exists(OUTPUT_FILE):
        sz = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        print(f"✅ ppe.pt already exists ({sz:.1f} MB) — nothing to do.")
        print("   Restart the backend to use it.")
        sys.exit(0)

    success = download_with_gdown()

    if not success:
        print("\n⚠️  Automatic download failed.")
        manual_instructions()
        sys.exit(1)

    print("\n🚀 Next step: restart the backend server.")
    print("   The startup log should show:")
    print("   ✅ Custom ppe.pt model loaded (PPE-native, 10 classes)")

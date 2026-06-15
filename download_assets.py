#!/usr/bin/env python3
"""
download_assets.py — fetch the trained model (and songs) that are NOT stored in git.

Why this exists
---------------
The trained model and the music library are distributed as **GitHub Release assets**
instead of being committed to the repository. That keeps the repo small and, crucially,
means a plain `git clone` OR a "Download ZIP" both work — no Git LFS, no broken pointer
files. This script downloads those assets into ./outputs and ./suno_ai_songs, placed
right next to this file (so it works no matter where you cloned the repo).

Usage
-----
    python3 download_assets.py

Requires only the Python standard library (no pip installs).
"""

import os
import sys
import json
import io
import zipfile
import urllib.request
import urllib.error

# ---- where the assets live: a GitHub Release on this repo --------------------
REPO = "ArjanWalia/Music-and-AI-for-Stress-Management"
TAG  = "assets-v1"          # <-- the Release tag you upload the files under
BASE = f"https://github.com/{REPO}/releases/download/{TAG}"

# ---- destinations: relative to THIS script (works in any clone) --------------
HERE   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(HERE, "outputs")
MUSIC  = os.path.join(HERE, "suno_ai_songs")

MODEL_FILES = ["fusionnet.pt", "feature_scaler.npz", "model_meta.json"]
SONGS_ZIP   = "suno_ai_songs.zip"   # a zip whose ROOT contains calm/ neutral/ elevated/


def _download(url, dest):
    print(f"  - {os.path.basename(dest)} ... ", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"done ({os.path.getsize(dest):,} bytes)")
        return True
    except urllib.error.HTTPError as e:
        print(f"FAILED (HTTP {e.code}) — is the Release tag '{TAG}' published with this asset?")
    except Exception as e:
        print(f"FAILED ({e})")
    return False


def main():
    print(f"Fetching assets from  {BASE}\n")

    # --- model files -> ./outputs ---
    os.makedirs(OUT, exist_ok=True)
    print("Model files -> outputs/")
    got_model = all(_download(f"{BASE}/{f}", os.path.join(OUT, f)) for f in MODEL_FILES)

    # --- songs (optional) -> ./suno_ai_songs ---
    print("\nSongs -> suno_ai_songs/")
    os.makedirs(MUSIC, exist_ok=True)
    try:
        print(f"  - {SONGS_ZIP} ... ", end="", flush=True)
        data = urllib.request.urlopen(f"{BASE}/{SONGS_ZIP}").read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(MUSIC)
        n = sum(len(files) for _, _, files in os.walk(MUSIC))
        print(f"extracted ({n} files)")
    except Exception as e:
        print(f"skipped ({e})")
        print("    (local music will be empty until songs are added; the app still runs in LOCAL mode without them)")

    # --- verify the model actually loaded as real files (not pointers/404s) ---
    print("\nVerifying model files:")
    ok = True
    mp = os.path.join(OUT, "model_meta.json")
    try:
        json.load(open(mp)); print("  model_meta.json : valid JSON  [OK]")
    except Exception as e:
        print(f"  model_meta.json : NOT valid JSON  [FAIL] ({e})"); ok = False
    for f, floor in (("fusionnet.pt", 10_000), ("feature_scaler.npz", 200)):
        p = os.path.join(OUT, f); sz = os.path.getsize(p) if os.path.isfile(p) else 0
        flag = "OK" if sz > floor else "FAIL (looks like a pointer or is missing)"
        print(f"  {f:18}: {sz:,} bytes  [{flag}]")
        ok &= sz > floor

    if ok:
        print("\nAll set. Run:  python3 server.py")
    else:
        print("\nSomething's off. Check that the Release tag and asset filenames match this script.")
        sys.exit(1)


if __name__ == "__main__":
    main()

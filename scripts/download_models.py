"""
scripts/download_models.py
Downloads trained model weights from Hugging Face Hub or checks local cache.
"""

import os
import shutil
import sys
import urllib.request

HF_BASE = os.getenv(
    "HF_MODEL_URL",
    "https://huggingface.co/Sahil-aka/deepfake-detection/resolve/main"
)

TARGET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_FILENAME = "model_87_acc_20_frames_final_data.pt"
TARGET_PATH = os.path.join(TARGET_DIR, MODEL_FILENAME)

# Known local fallback paths (e.g. parent workspace folder)
LOCAL_FALLBACKS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), MODEL_FILENAME),
    os.path.join(os.getcwd(), MODEL_FILENAME),
    os.path.join(os.getcwd(), "..", MODEL_FILENAME),
]


def download():
    os.makedirs(TARGET_DIR, exist_ok=True)

    if os.path.exists(TARGET_PATH) and os.path.getsize(TARGET_PATH) > 1000000:
        size_mb = os.path.getsize(TARGET_PATH) / (1024 * 1024)
        print(f"✅ Model weight already present: {TARGET_PATH} ({size_mb:.1f} MB)")
        return TARGET_PATH

    # Check local fallbacks first
    for fallback in LOCAL_FALLBACKS:
        if os.path.exists(fallback) and os.path.getsize(fallback) > 1000000:
            print(f"📦 Found local model at: {fallback}")
            print(f"📋 Copying to: {TARGET_PATH}")
            shutil.copyfile(fallback, TARGET_PATH)
            size_mb = os.path.getsize(TARGET_PATH) / (1024 * 1024)
            print(f"✅ Successfully copied model weights ({size_mb:.1f} MB)")
            return TARGET_PATH

    # Download from remote Hugging Face repo
    url = f"{HF_BASE}/{MODEL_FILENAME}"
    print(f"🌐 Downloading {MODEL_FILENAME} from {url}...")
    try:
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100.0, downloaded * 100.0 / total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\rDownloading: {percent:.1f}% ({mb:.1f}/{total_mb:.1f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, TARGET_PATH, reporthook=report_progress)
        print(f"\n✅ Saved model to {TARGET_PATH}")
        return TARGET_PATH
    except Exception as e:
        print(f"\n⚠️  Could not download from Hugging Face ({e}).")
        print("💡 You can manually copy 'model_87_acc_20_frames_final_data.pt' into 'models/' directory.")
        return None


if __name__ == "__main__":
    print("🔍 Checking DeepFake Detection pretrained models...")
    result = download()
    if result:
        print("🎉 Ready to run: uvicorn app:app --reload")
    else:
        print("⚠️ Model not found. Please provide the .pt file.")

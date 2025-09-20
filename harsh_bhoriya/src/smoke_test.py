# src/smoke_test.py
import sys
def check_imports():
    errors = []
    try:
        import ultralytics
    except Exception as e:
        errors.append(f"ultralytics import error: {e}")
    try:
        import cv2
    except Exception as e:
        errors.append(f"opencv-python import error: {e}")
    try:
        import torch
    except Exception as e:
        errors.append(f"torch import error: {e}")

    if errors:
        print("SMOKE TEST FAILED:")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    else:
        print("SMOKE TEST OK: ultralytics, opencv, torch imported successfully.")

if __name__ == "__main__":
    check_imports()

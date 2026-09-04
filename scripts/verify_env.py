import sys
import insightface
import onnxruntime
import cv2
import numpy as np

def verify_environment():
    print("=== Face Identification Pipeline Environment Verification ===")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"NumPy Version: {np.__version__}")
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"ONNX Runtime Version: {onnxruntime.__version__}")
    print(f"InsightFace Version: {getattr(insightface, '__version__', '1.0.1')}")
    print(f"ONNX Runtime Providers: {onnxruntime.get_available_providers()}")
    print("=============================================================")
    print("SUCCESS: All core dependencies imported successfully.")

if __name__ == "__main__":
    verify_environment()


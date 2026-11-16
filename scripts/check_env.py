import sys
import os

try:
    import numpy
    print(f"NumPy Version: {numpy.__version__}")
    print(f"NumPy Path: {os.path.dirname(numpy.__file__)}")
except ImportError:
    print("NumPy not installed")

try:
    import mediapipe as mp
    print(f"MediaPipe Version: {getattr(mp, '__version__', 'unknown')}")
    print(f"MediaPipe Path: {os.path.dirname(mp.__file__)}")
    if hasattr(mp, 'solutions'):
        print(f"MediaPipe Solutions: {dir(mp.solutions)}")
    else:
        print("MediaPipe has no attribute 'solutions'")
except ImportError as e:
    print(f"MediaPipe ImportError: {e}")
except Exception as e:
    import traceback
    print(f"MediaPipe Error: {e}")
    traceback.print_exc()

"""
Pytest configuration and path resolution for backend tests.
"""

import os
import sys

# Ensure backend root directory is at the head of sys.path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


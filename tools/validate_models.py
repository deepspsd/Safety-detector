#!/usr/bin/env python3
"""
Standalone CLI launcher for OccuSafe Pre-Live Model Validation Pipeline.
Calls backend/tests/validate_models.py and saves output in backend/tests/validation_results.
"""
import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TOOLS_DIR)
_BACKEND_DIR = os.path.join(_ROOT_DIR, "backend")

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from backend.tests.validate_models import main

if __name__ == "__main__":
    main()

"""Root CLI entrypoint proxy for JARVIS. Dispatches directly to backend/main.py."""
import sys
import os
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Switch working directory to backend so relative paths resolve cleanly
os.chdir(backend_dir)

from main import main

if __name__ == "__main__":
    sys.exit(main())

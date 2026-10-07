import sys
from pathlib import Path

# Add project root directory to sys.path so app and analysis modules can be imported
file_dir = Path(__file__).resolve().parent
parent_dir = file_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from app import app

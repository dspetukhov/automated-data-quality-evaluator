import sys
from pathlib import Path

# Allow bare imports like `from preprocess import ...` used in main.py
sys.path.insert(0, str(Path(__file__).parent.parent))

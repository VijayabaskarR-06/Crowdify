import os
import sys

# Add the root directory to the path so app.py can find monitor.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import app

"""
Central configuration for the project — all paths and constants live here only.

"""
import os 

# --- model path ---
MODEL_PATH = ""

# --- Class name prefixes considered a "violation" ---
VIOLATION_PREFIXES = ("no ", "no-")

# --- Output file paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOG_FILE = os.path.join(BASE_DIR, "violations_log.csv")
 
os.makedirs(REPORTS_DIR, exist_ok=True)

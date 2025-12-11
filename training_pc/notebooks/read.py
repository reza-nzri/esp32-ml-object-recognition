import os
import glob
import pandas as pd


# Automatically select the latest file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FOLDER = os.path.join(BASE_DIR, "data", "raw")

files = glob.glob(os.path.join(RAW_FOLDER, "*"))

if not files:
    raise FileNotFoundError("Raw folder empty.")

log_path = max(files, key=os.path.getmtime)
print("Loading newest file:", log_path)

# Read the CSV
df = pd.read_csv(log_path, comment="#")
print(df.head())

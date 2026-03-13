import os, json

# Try both possible locations
paths = [
    "data/raw/cuad/train.json",
    "data/raw/cuad_raw.json",
    "data/raw/CUAD_v1.json"
]

for path in paths:
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024*1024)
        print(f"Found: {path}  ({size:.2f} MB)")

print("\n--- Checking data/raw/ folder structure ---")
for root, dirs, files in os.walk("data/raw"):
    level = root.replace("data/raw", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    for file in files:
        fpath = os.path.join(root, file)
        fsize = os.path.getsize(fpath) / (1024*1024)
        print(f"{indent}  {file}  ({fsize:.2f} MB)")
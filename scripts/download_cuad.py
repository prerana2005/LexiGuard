import requests
import json
import os

os.makedirs("data/raw", exist_ok=True)

# Official CUAD dataset from GitHub release (no auth needed)
url = "https://github.com/TheAtticusProject/cuad/raw/main/data/master_clauses.json"

print("📥 Downloading CUAD from official GitHub...")
print("⏳ ~52MB, please wait...\n")

response = requests.get(url, stream=True, timeout=120)
response.raise_for_status()

total_bytes = 0
with open("data/raw/cuad_master.json", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
        total_bytes += len(chunk)
        mb = total_bytes / (1024 * 1024)
        print(f"  Downloaded: {mb:.1f} MB", end="\r")

print(f"\n✅ Download complete! {total_bytes/(1024*1024):.2f} MB")

# Verify and preview
print("\n🔍 Parsing downloaded file...")
with open("data/raw/cuad_master.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Type     : {type(data)}")

if isinstance(data, list):
    print(f"Records  : {len(data)}")
    print(f"Keys     : {list(data[0].keys())}")
    print(f"Sample   : {str(data[0])[:200]}")
elif isinstance(data, dict):
    print(f"Top keys : {list(data.keys())[:5]}")
    # SQuAD format
    if "data" in data:
        contracts = data["data"]
        print(f"Contracts: {len(contracts)}")
        total_qa = sum(len(p['qas'])
                      for doc in contracts
                      for p in doc['paragraphs'])
        print(f"QA pairs : {total_qa}")
        print(f"Sample title: {contracts[0]['title'][:80]}")

print("\n✅ CUAD data ready!")
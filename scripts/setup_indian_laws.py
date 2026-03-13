import os
import json

os.makedirs("data/indian_laws", exist_ok=True)

# We'll create placeholder files for each law
# You'll fill these with real content in the next step
laws = [
    "Indian_Contract_Act_1872",
    "DPDP_Act_2023",
    "Labour_Codes_2020",
    "RERA_2016",
    "IT_Act_2000",
    "Consumer_Protection_Act_2019",
    "Arbitration_Act_1996"
]

for law in laws:
    path = f"data/indian_laws/{law}.txt"
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"[PLACEHOLDER] {law}\n")
        print(f"Created: {path}")
    else:
        size = os.path.getsize(path)
        print(f"Exists : {path} ({size} bytes)")

print("\n✅ Indian laws folder ready!")
print("📁 Files in data/indian_laws/:")
for f in os.listdir("data/indian_laws"):
    print(f"   {f}")
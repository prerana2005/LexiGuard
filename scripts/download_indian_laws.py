import requests
from bs4 import BeautifulSoup
import time
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
}

os.makedirs("data/indian_laws", exist_ok=True)

# Wikipedia sources — reliable, no blocking
WIKI_LAWS = {
    "IT_Act_2000": "https://en.wikipedia.org/wiki/Information_Technology_Act,_2000",
    "Arbitration_Act_1996": "https://en.wikipedia.org/wiki/Arbitration_and_Conciliation_Act,_1996",
    "Consumer_Protection_Act_2019": "https://en.wikipedia.org/wiki/Consumer_Protection_Act,_2019",
    "RERA_2016": "https://en.wikipedia.org/wiki/Real_Estate_(Regulation_and_Development)_Act,_2016",
    "DPDP_Act_2023": "https://en.wikipedia.org/wiki/Digital_Personal_Data_Protection_Act,_2023",
    "Labour_Codes_2020": "https://en.wikipedia.org/wiki/Labour_codes_(India)",
    "Indian_Contract_Act_1872": "https://en.wikipedia.org/wiki/Indian_Contract_Act,_1872",
}

def scrape_wikipedia(law_name, url):
    out_path = f"data/indian_laws/{law_name}.txt"
    print(f"\n📥 Downloading: {law_name}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove unwanted sections
        for tag in soup.find_all(["table", "sup", "span"]):
            tag.decompose()

        # Get main content div
        content = soup.find("div", {"id": "mw-content-text"})
        if not content:
            print(f"   ⚠️ No content found")
            return 0

        # Extract all paragraphs and headings
        lines = []
        for tag in content.find_all(["h1","h2","h3","h4","p","li"]):
            text = tag.get_text(strip=True)
            if len(text) > 30:
                if tag.name in ["h1","h2","h3","h4"]:
                    lines.append(f"\n\n{'='*40}\n{text}\n{'='*40}")
                else:
                    lines.append(text)

        full_text = "\n\n".join(lines)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"LAW: {law_name}\n")
            f.write(f"SOURCE: {url}\n")
            f.write("="*60 + "\n\n")
            f.write(full_text)

        size = os.path.getsize(out_path) / 1024
        print(f"   ✅ Saved {size:.1f} KB")
        return size

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0

# Download all
results = {}
for law_name, url in WIKI_LAWS.items():
    size = scrape_wikipedia(law_name, url)
    results[law_name] = size
    time.sleep(1)

print("\n\n📊 Final Summary:")
for law, size in results.items():
    status = "✅" if size > 10 else "⚠️  Too small"
    print(f"   {status} {law}: {size:.1f} KB")
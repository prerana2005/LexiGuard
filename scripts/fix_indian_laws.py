import requests
from bs4 import BeautifulSoup
import time
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
}

# Fixed URLs for the 4 problem laws
FIX_LAWS = {
    "Arbitration_Act_1996": "https://en.wikipedia.org/wiki/Arbitration_and_Conciliation_Act_1996",
    "Labour_Codes_2020": "https://en.wikipedia.org/wiki/Labour_codes_in_India",
    "Consumer_Protection_Act_2019": "https://en.wikipedia.org/wiki/Consumer_Protection_Act_2019",
    "RERA_2016": "https://en.wikipedia.org/wiki/Real_Estate_Regulation_and_Development_Act_2016",
}

def scrape_wikipedia(law_name, url):
    out_path = f"data/indian_laws/{law_name}.txt"
    print(f"\n📥 Fetching: {law_name}")
    print(f"   URL: {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup.find_all(["table", "sup"]):
            tag.decompose()

        content = soup.find("div", {"id": "mw-content-text"})
        if not content:
            print(f"   ⚠️ No content div found")
            return 0

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

    except requests.exceptions.HTTPError as e:
        print(f"   ❌ HTTP Error: {e}")
        # Try searching Wikipedia for correct URL
        search_url = f"https://en.wikipedia.org/w/index.php?search={law_name.replace('_', '+')}"
        print(f"   🔍 Try manually: {search_url}")
        return 0
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0

results = {}
for law_name, url in FIX_LAWS.items():
    size = scrape_wikipedia(law_name, url)
    results[law_name] = size
    time.sleep(1)

print("\n\n📊 Fix Summary:")
for law, size in results.items():
    status = "✅" if size > 5 else "⚠️  Still missing"
    print(f"   {status} {law}: {size:.1f} KB")

print("\n📊 All Laws Status:")
for f in os.listdir("data/indian_laws"):
    path = f"data/indian_laws/{f}"
    size = os.path.getsize(path) / 1024
    status = "✅" if size > 5 else "❌"
    print(f"   {status} {f}: {size:.1f} KB")
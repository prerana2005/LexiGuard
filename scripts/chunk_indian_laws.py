import os
import json
import re

def split_into_chunks(text, max_tokens=400):
    """Split text into chunks of approximately max_tokens words."""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)

        if current_length + word_count > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = words
            current_length = word_count
        else:
            current_chunk.extend(words)
            current_length += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def extract_section_info(chunk_text):
    """Try to extract section number and topic from chunk text."""
    # Look for patterns like "Section 10 - Title" or "Section 10."
    section_match = re.search(
        r'[Ss]ection\s+(\d+[A-Z]?)\s*[-–—]?\s*([^\n]{0,60})', 
        chunk_text
    )
    if section_match:
        sec_num = f"Section {section_match.group(1)}"
        topic   = section_match.group(2).strip()[:50]
        return sec_num, topic

    # Look for "Chapter" patterns
    chapter_match = re.search(
        r'[Cc]hapter\s+([IVXLC\d]+)\s*[-–—]?\s*([^\n]{0,60})',
        chunk_text
    )
    if chapter_match:
        sec_num = f"Chapter {chapter_match.group(1)}"
        topic   = chapter_match.group(2).strip()[:50]
        return sec_num, topic

    return "General", "General Provision"

# Process all law files
law_dir    = "data/indian_laws"
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

all_chunks = []
summary    = {}

print("📂 Processing Indian law files...\n")

for filename in sorted(os.listdir(law_dir)):
    if not filename.endswith(".txt"):
        continue

    law_name = filename.replace(".txt", "")
    filepath = os.path.join(law_dir, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Remove header lines (first 3 lines are metadata)
    lines      = text.split("\n")
    clean_text = "\n".join(lines[3:]).strip()

    # Split into chunks
    chunks = split_into_chunks(clean_text, max_tokens=400)

    law_chunks = []
    for i, chunk_text in enumerate(chunks):
        if len(chunk_text.strip()) < 50:
            continue  # Skip very short chunks

        section_num, topic = extract_section_info(chunk_text)

        chunk = {
            "law_name"      : law_name,
            "section_number": section_num,
            "topic"         : topic,
            "chunk_id"      : i,
            "text"          : chunk_text.strip()
        }
        law_chunks.append(chunk)
        all_chunks.append(chunk)

    summary[law_name] = len(law_chunks)
    print(f"  ✅ {law_name}")
    print(f"     Chunks created : {len(law_chunks)}")
    print(f"     Sample section : {law_chunks[0]['section_number']}")
    print(f"     Sample topic   : {law_chunks[0]['topic']}")
    print()

# Save all chunks
out_path = os.path.join(output_dir, "indian_laws_chunks.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print("="*55)
print(f"✅ Total chunks created : {len(all_chunks)}")
print(f"💾 Saved to            : {out_path}")
print("\n📊 Chunks per law:")
for law, count in summary.items():
    print(f"   {law}: {count} chunks")

# Preview one full chunk
print("\n📄 Sample chunk:")
sample = all_chunks[5]
for k, v in sample.items():
    print(f"   {k:15}: {str(v)[:100]}")
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the embedding model (same one Vineet used)
print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load the FAISS index
print("Loading FAISS index...")
index = faiss.read_index('faiss_index/indian_laws.index')

# Load the metadata (law name, section, text)
print("Loading metadata...")
with open('faiss_index/indian_laws_metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

print(f"FAISS index loaded! Total law chunks: {index.ntotal}")

def search_relevant_laws(clause_text, top_k=3):
    # Convert clause to vector
    clause_vector = model.encode([clause_text])
    clause_vector = np.array(clause_vector).astype('float32')

    # Search more results so we can filter bad ones
    distances, indices = index.search(clause_vector, top_k * 5)

    # Get results but FILTER OUT generic QA entries
    results = []
    bad_law_names = ['Indian_Law_QA', 'Indian_Law', 'Indian_Legal']
    
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            chunk = metadata[idx]
            law_name = chunk.get('law_name', 'Unknown')
            
            # Skip bad entries
            if law_name in bad_law_names:
                continue
                
            results.append({
                'law_name': law_name,
                'section_number': chunk.get('section_number', 'Unknown'),
                'text': chunk.get('text', ''),
                'relevance_score': float(distances[0][i])
            })
            
            # Stop once we have enough good results
            if len(results) == top_k:
                break

    return results


# Test it with a sample clause
if __name__ == "__main__":
    test_clause = "The company may terminate the employee at any time without prior notice or reason."
    
    print(f"\nSearching laws for clause:\n'{test_clause}'\n")
    results = search_relevant_laws(test_clause)
    
    print("Top 3 relevant Indian law chunks found:")
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Law: {result['law_name']}")
        print(f"Section: {result['section_number']}")
        print(f"Text: {result['text'][:200]}...")
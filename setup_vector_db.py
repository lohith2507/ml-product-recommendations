import sqlite3
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = os.path.join(os.path.dirname(__file__), "recommendation.db")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "products_text.index")
MAP_PATH = os.path.join(os.path.dirname(__file__), "products_vector_map.json")

def setup_vector_db():
    print("Loading embedding model (this may take a moment to download)...")
    # all-MiniLM-L6-v2 is fast and produces good embeddings (384 dimensions)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    products = conn.execute("SELECT product_id, title, description, category, features FROM products").fetchall()
    
    if not products:
        print("No products found in the database. Run setup_data.py first.")
        return
        
    print(f"Embedding {len(products)} products...")
    
    texts = []
    product_ids = []
    
    for row in products:
        product_id = row['product_id']
        title = row['title'] or ""
        desc = row['description'] or ""
        category = row['category'] or ""
        
        try:
            features = json.loads(row['features']) if isinstance(row['features'], str) else []
            features_text = " ".join(features)
        except:
            features_text = ""
            
        # Combine rich text for semantic search
        combined_text = f"{title}. {category}. {desc}. {features_text}"
        texts.append(combined_text)
        product_ids.append(product_id)
        
    # Generate embeddings
    print("Generating embeddings... (may take a minute or two)")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1] # Should be 384
    index = faiss.IndexFlatL2(dimension)
    
    # Normalize for cosine similarity instead of L2 distance (optional but better for text)
    faiss.normalize_L2(embeddings)
    
    index.add(embeddings)
    
    # Save the index
    print(f"Saving FAISS index to {INDEX_PATH}...")
    faiss.write_index(index, INDEX_PATH)
    
    # Save the mapping from FAISS index ID -> product_id
    # FAISS uses 0-based integer IDs which correspond to our array indices
    mapping = {i: pid for i, pid in enumerate(product_ids)}
    with open(MAP_PATH, "w") as f:
        json.dump(mapping, f)
        
    print(f"[OK] Vector DB created with {len(product_ids)} products.")

if __name__ == "__main__":
    setup_vector_db()

import os
import json
import faiss
from sentence_transformers import SentenceTransformer

DB_DIR = os.path.dirname(os.path.dirname(__file__))
INDEX_PATH = os.path.join(DB_DIR, "products_text.index")
MAP_PATH = os.path.join(DB_DIR, "products_vector_map.json")

class SemanticSearch:
    def __init__(self):
        self.model = None
        self.index = None
        self.mapping = None
        
    def _load_resources(self):
        if self.model is None:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        if self.index is None and os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
        if self.mapping is None and os.path.exists(MAP_PATH):
            with open(MAP_PATH, "r") as f:
                # keys in JSON are strings, convert to int
                self.mapping = {int(k): v for k, v in json.load(f).items()}
                
    def search(self, query: str, top_k: int = 10):
        self._load_resources()
        if not self.index or not self.mapping:
            return []
            
        # Encode query
        query_vector = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        
        # Search
        distances, indices = self.index.search(query_vector, top_k)
        
        # Map back to product IDs
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.mapping:
                results.append({
                    "product_id": self.mapping[idx],
                    "similarity": float(distances[0][i])
                })
                
        return results

semantic_engine = SemanticSearch()

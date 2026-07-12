import sqlite3
import json
import os
import random

DB_PATH = os.path.join(os.path.dirname(__file__), "recommendation.db")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "finetune_dataset.jsonl")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def generate_dataset():
    print(f"Generating fine-tuning dataset to {OUTPUT_PATH}...")
    
    with get_connection() as conn:
        # Get users and their preferences
        users = conn.execute("SELECT user_id, preferences FROM users").fetchall()
        user_prefs = {}
        for u in users:
            try:
                prefs = json.loads(u["preferences"]) if isinstance(u["preferences"], str) else {}
                user_prefs[u["user_id"]] = prefs
            except json.JSONDecodeError:
                user_prefs[u["user_id"]] = {}

        # Get highly rated reviews (4.0 and above) to form positive recommendations
        # We join with products to get product details
        query = """
            SELECT r.user_id, p.title, p.category, p.description, p.price, p.avg_rating
            FROM reviews r
            JOIN products p ON r.product_id = p.product_id
            WHERE r.rating >= 4.0
            LIMIT 5000
        """
        highly_rated_interactions = conn.execute(query).fetchall()
        
    dataset = []
    
    system_prompt = "You are a friendly and knowledgeable AI shopping assistant for an e-commerce platform. You help users discover products based on their needs and preferences."
    
    for interaction in highly_rated_interactions:
        user_id = interaction["user_id"]
        title = interaction["title"]
        category = interaction["category"]
        desc = interaction["description"] or "A fantastic product."
        price = interaction["price"]
        rating = interaction["avg_rating"]
        
        prefs = user_prefs.get(user_id, {})
        liked_categories = prefs.get("liked_categories", [])
        
        # Construct human query
        query_templates = [
            f"I am looking for something in {category}.",
            f"Can you suggest a good {category} product?",
            f"I need a recommendation for {category}. I prefer high-rated items.",
            f"What's a good product you recommend in the {category} department?"
        ]
        
        if liked_categories and category in liked_categories:
            query_templates.append(f"Since I like {category}, what would you recommend?")
            
        human_query = random.choice(query_templates)
        
        # Construct assistant response
        desc_preview = desc[:150] + "..." if len(desc) > 150 else desc
        response_templates = [
            f"I highly recommend the **{title}**. It's highly rated at {rating} stars and costs ${price:.2f}. {desc_preview}",
            f"Based on what you're looking for, the **{title}** is an excellent choice. Customers give it {rating} stars! Here's a bit about it: {desc_preview}",
            f"You should check out the **{title}**. At ${price:.2f} with a {rating} star rating, it's a great option in {category}.",
            f"I've got the perfect suggestion: **{title}**. It fits your needs perfectly. {desc_preview}"
        ]
        
        gpt_response = random.choice(response_templates)
        
        # Create ShareGPT formatted conversation
        conversation = {
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": human_query},
                {"from": "gpt", "value": gpt_response}
            ]
        }
        dataset.append(conversation)
        
    # Write to JSONL
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
            
    print(f"[OK] Generated {len(dataset)} examples for fine-tuning!")
    print("Format: ShareGPT (JSONL)")

if __name__ == "__main__":
    generate_dataset()

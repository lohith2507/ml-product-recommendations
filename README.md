# 🧠 AI-Based Product Recommendation System

An AI-powered e-commerce recommendation engine combining **Collaborative Filtering** (Surprise SVD), **Content-Based Filtering** (TF-IDF), and **LLM-Powered Hybrid Recommendations** (Groq/Llama 3.3) — with a premium dark-mode web dashboard, analytics, and AI chatbot.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![Surprise](https://img.shields.io/badge/Surprise-SVD-purple)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3-orange)

## ✨ Features

- **🤝 Collaborative Filtering** — SVD matrix factorization with GridSearchCV hyperparameter tuning
- **📝 Content-Based Filtering** — TF-IDF vectorization with cosine similarity
- **🧠 AI Hybrid Recommender** — LLM re-ranking with personalized explanations via Groq API
- **🛒 "Also Bought" Recommendations** — Co-purchase pattern detection
- **💬 AI Shopping Chatbot** — Conversational product discovery powered by Llama 3.3
- **📊 Analytics Dashboard** — CTR tracking, algorithm comparison charts
- **🧪 A/B Testing** — Statistical significance testing between algorithms
- **🌙 Premium Dark UI** — Glassmorphism design with smooth animations

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment
Create a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Generate Data & Train Models
```bash
python setup_data.py
python train_models.py
```

### 4. Start the Server
```bash
uvicorn app:app --reload --port 8000
```

### 5. Open Dashboard
Visit [http://localhost:8000](http://localhost:8000)

API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

## 📂 Project Structure

```
ml-product-recommendations/
├── app.py                    # FastAPI main application
├── setup_data.py             # Synthetic dataset generator
├── train_models.py           # Model training orchestrator
├── database/
│   ├── models.py             # SQLite schema & connection manager
│   └── __init__.py
├── recommenders/
│   ├── collaborative.py      # Surprise SVD engine
│   ├── content_based.py      # TF-IDF engine
│   ├── llm_hybrid.py         # Groq LLM hybrid engine
│   ├── ab_testing.py         # A/B testing engine
│   └── __init__.py
├── api/
│   ├── routes.py             # REST API endpoints
│   ├── schemas.py            # Pydantic models
│   └── __init__.py
├── static/
│   ├── css/styles.css        # Premium dark glassmorphism UI
│   └── js/
│       ├── app.js            # Main app logic
│       ├── recommendations.js # Product card rendering
│       ├── chatbot.js        # AI chatbot widget
│       └── analytics.js      # Chart.js analytics
└── templates/
    └── index.html            # Dashboard template
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/users` | List users |
| `GET` | `/api/users/{id}` | User profile |
| `GET` | `/api/products` | Browse products |
| `GET` | `/api/recommend/collaborative/{user_id}` | CF recommendations |
| `GET` | `/api/recommend/content/{user_id}` | CB recommendations |
| `GET` | `/api/recommend/hybrid/{user_id}` | LLM hybrid recommendations |
| `GET` | `/api/recommend/also-bought/{product_id}` | Co-purchase recommendations |
| `POST` | `/api/chat` | AI chatbot |
| `GET` | `/api/analytics/ctr` | CTR analytics |
| `GET` | `/api/analytics/ab-test` | A/B test results |

## 🏗️ Architecture

- **Data Layer**: SQLite with synthetic e-commerce data (500 products, 200 users, ~5K reviews)
- **CF Engine**: Surprise SVD with GridSearchCV tuning (RMSE < 1.0)
- **CB Engine**: TF-IDF + Cosine Similarity on product text (weighted: category 3x, title 2x)
- **LLM Engine**: Groq API (Llama 3.3 70B) for re-ranking + natural language explanations
- **A/B Testing**: Chi-squared significance testing across 4 algorithm groups

## 📄 License

MIT License

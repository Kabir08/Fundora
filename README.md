---
title: Fundora
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: AI-powered scholarship matcher — upload your CV, get matched
---

# 🎓 Fundora — AI Scholarship Matcher

Find scholarships that match your profile from a curated index of 90+ global programmes — Chevening, DAAD, Fulbright, Google PhD Fellowship, MEXT, Vanier, Mastercard Foundation, and more.

## What's inside

| URL | What it does |
|-----|---|
| `/ui` | **Gradio web app** — upload resume or paste profile, get ranked scholarships |
| `/match` | **ChatGPT Action API** — JSON endpoint for the custom GPT |
| `/health` | Keep-alive ping endpoint |
| `/docs` | FastAPI Swagger UI |
| `/privacy` | Privacy policy (required by ChatGPT Actions) |

## Two search modes

### ⚡ Quick Match *(recommended)*
Uses a pre-loaded index of 90+ hand-curated scholarships. Results in seconds. No scraping.

### 🔍 Custom Search
Paste any scholarship listing URLs — Fundora scrapes and ranks them against your profile on the fly.

## Tech stack
- **FastAPI** — REST API + ChatGPT Action backend  
- **Gradio** — direct web UI  
- **sentence-transformers** (`all-MiniLM-L6-v2`) — semantic embeddings  
- **ChromaDB** — persistent vector store (seed loaded at startup)  
- **FAISS** — in-memory ranking for custom URL searches  
- **httpx** — lightweight HTTP scraping (no browser required)

## Local dev

```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 7860
# Open http://localhost:7860/ui
```

## ChatGPT Action setup

After deploying to HuggingFace Spaces, update `openapi_schema.json`:
```json
"servers": [{ "url": "https://kabir08-fundora.hf.space" }]
```
Then re-import the schema in your GPT configuration.

## GitHub
[github.com/Kabir08/Fundora](https://github.com/Kabir08/Fundora)

---

This README describes the HuggingFace Space tool for matching user profiles to scraped opportunities. See `app.py` for implementation details.

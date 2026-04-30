---
title: Fundora
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: AI scholarship matcher — upload CV, get ranked results
---

<div align="center">

# 🎓 Fundora

**AI-powered scholarship matcher — upload your CV, get personalised ranked results**

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace%20Space-Live%20Demo-blue?style=for-the-badge)](https://huggingface.co/spaces/k01010/fundora)
[![API Docs](https://img.shields.io/badge/FastAPI-Swagger%20Docs-009688?style=for-the-badge&logo=fastapi)](https://k01010-fundora.hf.space/docs)
[![GitHub](https://img.shields.io/badge/GitHub-Kabir08%2FFundora-181717?style=for-the-badge&logo=github)](https://github.com/Kabir08/Fundora)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange?logo=gradio)
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-2.2+-yellow)
![HuggingFace Spaces](https://img.shields.io/badge/Hosted%20on-HuggingFace%20Spaces-FFD21E?logo=huggingface)

</div>

---

## What is Fundora?

Fundora is a **semantic scholarship search engine**. You describe yourself (or upload your resume/CV), and Fundora ranks the best-matching global scholarships for you — using sentence-level AI embeddings, not keyword matching.

It runs as a **web app** (Gradio UI at `/ui`) and simultaneously as a **ChatGPT Action API** (at `/match`), so it can be integrated directly into a custom ChatGPT GPT.

---

## 🚀 Try it live

| Surface | URL |
|---|---|
| **Web App (Gradio UI)** | [https://k01010-fundora.hf.space/ui](https://k01010-fundora.hf.space/ui) |
| **Swagger API Docs** | [https://k01010-fundora.hf.space/docs](https://k01010-fundora.hf.space/docs) |
| **Health Check** | [https://k01010-fundora.hf.space/health](https://k01010-fundora.hf.space/health) |
| **Privacy Policy** | [https://k01010-fundora.hf.space/privacy](https://k01010-fundora.hf.space/privacy) |

---

## ✨ Features

- **Upload your CV** (PDF, DOCX, or TXT) or paste a plain-text profile
- **Semantic matching** using `sentence-transformers/all-MiniLM-L6-v2` — goes beyond keyword search
- **93 hand-curated scholarships** loaded at startup (no scraping delay)
- **Custom URL search** — paste any scholarship listing page and Fundora scrapes + ranks on the fly
- **ChatGPT Action** — plug this into your own custom GPT
- **Zero browser dependency** — replaced Playwright with `httpx` for lightweight scraping
- **Persistent vector store** via ChromaDB on HuggingFace Spaces disk

---

## 🖥️ How to use the Web App

### ⚡ Quick Match *(recommended)*

1. Open [https://k01010-fundora.hf.space/ui](https://k01010-fundora.hf.space/ui)
2. Upload your resume (PDF/DOCX/TXT) **or** paste a text description of yourself in the box
3. Set how many results you want (default: 10)
4. Click **Find Scholarships**
5. Get a ranked table with scholarship name, link, and why it matches you

**Example profile text:**
```
Final year BE Computer Science student, GPA 8.5. Research interest in NLP and LLMs.
Published paper on RAG systems. Seeking fully funded MS or PhD scholarships in USA or Europe.
```

### 🔍 Custom Search

1. Switch to the **Custom Search** tab
2. Paste one or more scholarship listing URLs (one per line)
3. Add your profile text
4. Click **Search** — Fundora scrapes those pages and ranks results against your profile

---

## 🤖 ChatGPT Action Integration

Fundora exposes a `/match` endpoint that your custom ChatGPT GPT can call.

**Step-by-step:**
1. Go to [chat.openai.com](https://chat.openai.com) → **My GPTs** → Create/Edit GPT
2. Click **Add Action** → Import from URL:
   ```
   https://k01010-fundora.hf.space/openapi.json
   ```
   Or paste the contents of [`openapi_schema.json`](openapi_schema.json) manually
3. Set authentication to **None**
4. Save and test by asking: *"Find me scholarships for a computer science PhD student from India"*

**Request format:**
```json
POST /match
{
  "profile": "Your academic background, research interests, nationality, degree level...",
  "top_k": 5
}
```

**Response format:**
```json
{
  "matches": [
    {
      "title": "Google PhD Fellowship",
      "link": "https://research.google/outreach/phd-fellowship/",
      "score": 0.91,
      "description": "Google PhD Fellowship supports outstanding PhD students..."
    }
  ],
  "index_size": 93,
  "index_ready": true
}
```

---

## 📊 Scholarship Index — Statistics

### Overview

| Metric | Value |
|---|---|
| **Total scholarships indexed** | 93 |
| **Regions covered** | 12 |
| **Sources / organisations** | 60+ |
| **Index type** | ChromaDB cosine similarity |
| **Embedding model** | `all-MiniLM-L6-v2` (384-dim) |
| **Cold start time** | ~15–20 seconds (seed load) |
| **Query latency** | < 500ms |

### Breakdown by Region

| Region | Count | Example Scholarships |
|---|---|---|
| 🇺🇸 USA | 24 | NSF GRFP, Google PhD, Hertz, Ford Foundation, AAUW, MIT, CMU, Berkeley, Caltech, Stanford |
| 🇬🇧 UK / Europe | 16 | Chevening, Gates Cambridge, Rhodes, Commonwealth, DAAD, ETH Zurich, Erasmus Mundus, Marshall |
| 🤖 AI / CS Specific | 10 | DeepMind, OpenAI, Meta, Amazon, Apple, Jane Street, Two Sigma, Qualcomm, Snap |
| 🌍 Other / International | 9 | World Bank, ADB, Aga Khan, Rotary, UN, AfterSchoolAfrica |
| 🔗 Aggregator Platforms | 7 | Scholars4Dev, OpportunityDesk, FindAPhD, ScholarshipPortal, Niche, Bold.org, Going Merry |
| 🇮🇳 India | 7 | ICCR, INSPIRE, PMRF, Inlaks, Narotam Sekhsaria, S.N. Bose, NOS |
| 🇨🇦 Canada | 4 | Vanier, Trudeau Foundation, NSERC, Mitacs |
| 🌏 Asia | 6 | MEXT (Japan), JASSO, GKS (Korea), CSC (China), ASEAN, IIT |
| 🇦🇺 Australia | 4 | Australia Awards, Melbourne, ANU, Endeavour |
| 🏆 Leadership / Social Impact | 5 | Schwarzman, Knight-Hennessy, Paul & Daisy Soros, Mitchell, Marshall |
| 🌍 Africa | 3 | Mastercard Foundation, African Union, AfterSchoolAfrica |

### Full Scholarship List

<details>
<summary>Click to expand all 93 scholarships</summary>

**UK / Europe**
- Chevening Scholarship · Gates Cambridge Scholarship · Rhodes Scholarship
- Commonwealth Scholarship (Masters & PhD) · DAAD · Heinrich Böll Foundation
- Erasmus Mundus Joint Masters · UCL Graduate Scholarships · Edinburgh Global Research
- Oxford Clarendon Scholarship · ETH Zurich Excellence Scholarship
- Swiss Government Excellence Scholarships · EURAXESS EU Mobility · Marie Skłodowska-Curie (MSCA)
- Marshall Scholarship

**USA**
- NSF Graduate Research Fellowship (GRFP) · Google PhD Fellowship · Microsoft Research PhD Fellowship
- NVIDIA Graduate Fellowship · IBM PhD Fellowship · Adobe Research Fellowship
- Hertz Fellowship · Ford Foundation Predoctoral Fellowship · AAUW International Fellowships
- CMU University Fellowship · MIT Fellowship Programs · Berkeley Graduate Fellowship
- Stanford HCI Fellowships · Caltech Fellowship · DARPA Riser Award
- AISES Scholarship · SWE Scholarship · NSBE Scholarships · Hispanic Scholarship Fund
- Gates Scholarship (minority undergrad) · Fulbright Foreign Student · Simons Foundation Fellowship
- Siemens Foundation Scholarship · ACM-ICPC Scholarship

**AI / CS / Tech**
- OpenAI Scholars Program · OpenAI Residency · DeepMind Scholarship
- Meta AI Research Fellowship · Amazon Graduate Research Fellowship
- Apple Scholars in AI/ML · Jane Street Graduate Fellowship · Two Sigma PhD Fellowship
- Qualcomm Innovation Fellowship · Snap Research Fellowship

**Canada**
- Vanier Canada Graduate Scholarship · Pierre Elliott Trudeau Foundation Doctoral
- NSERC Postgraduate Scholarship · Mitacs Globalink Research Award

**Australia**
- Australia Awards Scholarships · University of Melbourne Graduate Research
- ANU Chancellor's International Scholarship · Endeavour Leadership Program

**Asia**
- MEXT Japanese Government Scholarship · JASSO Scholarship · Korean Government Scholarship (GKS)
- Chinese Government Scholarship (CSC) · ASEAN Undergraduate Scholarship (Singapore)
- IIT Scholarship for International Students

**India**
- ICCR Scholarship · INSPIRE · Prime Minister's Research Fellowship (PMRF)
- GyanDhan · National Overseas Scholarship (NOS) · Inlaks Shivdasani Foundation
- Narotam Sekhsaria Foundation · S.N. Bose Scholars Program · USC Viterbi India Program

**Africa**
- AfterSchoolAfrica Database · Mastercard Foundation Scholars · African Union Scholarships

**International Organisations**
- World Bank Graduate Scholarship (JJ/WBGSP) · ADB-Japan Scholarship
- Aga Khan Foundation International · Rotary Peace Fellowship · UN Scholarships & Fellowships

**Leadership / Social Impact**
- Schwarzman Scholars · Knight-Hennessy Scholars (Stanford) · Paul & Daisy Soros Fellowship
- Mitchell Scholarship (Ireland) · DAAD RISE

**Aggregator Platforms**
- Scholars4Dev · Opportunity Desk · FindAPhD · InternationalScholarships.com
- Niche No-Essay · Bold.org · Going Merry · ScholarshipPortal

</details>

---

## 🕸️ Scraping Sources (Live Index)

When no seed file is available, Fundora falls back to live-scraping **50+ sites**. These are also available for custom URL search:

| Category | Sites |
|---|---|
| Global aggregators | OpportunityDesk, Scholars4Dev, ScholarshipPortal, FindAPhD, InternationalScholarships, CareerFoundry |
| Europe | DAAD, EURAXESS, Erasmus+, Heinrich Böll, Chevening, Commonwealth, UCL, Oxford, Gates Cambridge, Rhodes |
| USA | Fulbright, Fastweb, CollegeBoard, Going Merry, Niche, Bold.org |
| Canada | EduCanada, Vanier, Trudeau Foundation, ScholarshipsCanada |
| Australia | Australia Awards, StudyInAustralia, ANU, University of Melbourne |
| Asia | MEXT, JASSO, GKS, CSC, Singapore MOE, ASEAN, GyanDhan |
| Africa/Middle East | AfterSchoolAfrica, African Union, Mastercard Foundation |
| India | ICCR, NOS, PMRF (via search) |

> **Note:** Many sites use bot-protection (Cloudflare). The seed index (`scholarships_seed.json`) is the reliable primary source and loads in ~15 seconds.

---

## 🛠️ Tech Stack

| Component | Library / Tool | Version |
|---|---|---|
| API framework | FastAPI | `>=0.100.0` |
| ASGI server | uvicorn[standard] | `>=0.20.0` |
| Web UI | Gradio | `>=4.0.0` |
| Embedding model | sentence-transformers (`all-MiniLM-L6-v2`) | `>=2.2.0` |
| Persistent vector DB | ChromaDB | `>=0.5.0` |
| In-memory ranking | faiss-cpu | `>=1.7.0` |
| HTML parsing | BeautifulSoup4 | `>=4.12.0` |
| HTTP scraping | httpx | `>=0.24.0` |
| PDF parsing | pdfplumber | `>=0.9.0` |
| DOCX parsing | python-docx | `>=0.8.11` |
| Data validation | pydantic | `>=2.0.0` |
| Hosting | HuggingFace Spaces (Docker) | — |

### Architecture

```
User uploads CV / types profile
         │
         ▼
  Gradio UI (/ui)  ◄────────────────── ChatGPT Action (/match)
         │                                      │
         ▼                                      ▼
   api.match()  ──────────────────────────────────
         │
         ▼
  sentence-transformers
  all-MiniLM-L6-v2 (384-dim embeddings)
         │
         ▼
  ChromaDB cosine similarity search
  (93-entry seed index, loaded at startup)
         │
         ▼
  Top-K ranked scholarships → returned to user
```

---

## 💻 Local Development

### Prerequisites
- Python 3.11+
- ~600 MB RAM (for the embedding model)

### Setup

```bash
git clone https://github.com/Kabir08/Fundora.git
cd Fundora
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --reload --port 7860
```

Open [http://localhost:7860/ui](http://localhost:7860/ui) in your browser.

### Regenerate the scholarship seed (optional)

```bash
python generate_seed.py
# Writes scholarships_seed.json
```

> Many sites block automated scrapers. The committed `scholarships_seed.json` is the reliable source.

### Docker (matches HuggingFace deployment exactly)

```bash
docker build -t fundora .
docker run -p 7860:7860 fundora
```

---

## 📦 Dependency Notes & Update Guide

When upgrading dependencies, these are the things to check:

| Package | Notes on upgrade |
|---|---|
| `gradio` | Breaking API changes between major versions (3→4→5). Check `gr.Blocks`, `gr.File`, `gr.Dataframe` signatures. |
| `chromadb` | Client API changed significantly at 0.4→0.5. If upgrading past 0.5, audit `Collection.add()`, `Collection.query()` calls in `api.py`. |
| `sentence-transformers` | Model download path uses `TRANSFORMERS_CACHE` / `HF_HOME`. Ensure env vars are set in Dockerfile. |
| `pydantic` | v1 → v2 is a breaking change. All models use `pydantic>=2.0`. Do not pin to v1. |
| `fastapi` | Follows pydantic — keep both in sync. |
| `faiss-cpu` | CPU-only build. Do not switch to `faiss-gpu` on HuggingFace free tier. |
| `httpx` | Used for all scraping. If upgrading past 1.x, check `httpx.Client` context manager usage in `api.py` and `app.py`. |
| `pdfplumber` | Used only in `app.py`'s `extract_text_from_file()`. Upgrade is safe. |

---

## 🗂️ Project Structure

```
Fundora/
├── api.py                  # FastAPI app — /match, /health, /index_status, /privacy
│                           #   + mounts Gradio at /ui
├── app.py                  # Gradio UI — Quick Match tab + Custom Search tab
├── scholarships_seed.json  # 93 hand-curated scholarships (loaded at startup)
├── generate_seed.py        # Script to regenerate scholarships_seed.json
├── openapi_schema.json     # ChatGPT Action schema (server URL: k01010-fundora.hf.space)
├── requirements.txt        # Python dependencies
├── Dockerfile              # HuggingFace Docker Space definition
├── render.yaml             # Legacy Render.com config (no longer primary host)
└── README.md               # This file
```

---

## 🔧 Deployment — HuggingFace Spaces

The app is deployed as a **Docker Space** at [https://huggingface.co/spaces/k01010/fundora](https://huggingface.co/spaces/k01010/fundora).

### Push updates

```bash
# One-time setup
git remote add hf https://huggingface.co/spaces/k01010/fundora

# Deploy
git add .
git commit -m "your message"
git push hf main        # deploys to HF Spaces
git push origin main    # keeps GitHub in sync
```

### Keep the Space awake

HuggingFace free Spaces sleep after 48 hours of inactivity. Set up a free ping via [UptimeRobot](https://uptimerobot.com):

- Monitor type: **HTTP(S)**
- URL: `https://k01010-fundora.hf.space/health`
- Interval: **every 14 minutes**

---

## 🔌 API Reference

### `POST /match`

Match a user profile against the scholarship index.

| Field | Type | Required | Description |
|---|---|---|---|
| `profile` | string | ✅ | Natural language description of the user |
| `top_k` | integer | ❌ | Number of results to return (default: 5) |

### `GET /health`

Returns `{"status": "ok", "index_ready": true/false}`. Use for keep-alive pings.

### `GET /index_status`

Returns current index size and readiness state.

### `POST /refresh_index`

Forces a re-scrape and re-index from the seed file.

### `GET /docs`

FastAPI auto-generated Swagger UI — try all endpoints interactively.

---

## 🤝 Contributing

1. Fork the repo
2. Add scholarships to `scholarships_seed.json` (follow the existing JSON schema)
3. Or improve scraping in `generate_seed.py`
4. Open a PR

**Scholarship JSON schema:**
```json
{
  "title": "Scholarship Name",
  "link": "https://official-url.com",
  "source": "OrganisationShortName",
  "description": "2–3 sentence description covering eligibility, value, and focus area."
}
```

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">
Built by <a href="https://github.com/Kabir08">Kabir08</a> · Hosted on <a href="https://huggingface.co/spaces/k01010/fundora">HuggingFace Spaces</a>
</div>

---

This README describes the HuggingFace Space tool for matching user profiles to scraped opportunities. See `app.py` for implementation details.

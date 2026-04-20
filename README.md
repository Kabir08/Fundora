# Scholarship & Opportunity Matcher

This HuggingFace Space lets users upload their background/profile file and input one or more URLs to scrape for opportunities (e.g., scholarships). It uses Firecrawl for scraping, MiniLM for embeddings, and FAISS for retrieval.

## Features
- Upload your profile/background file (TXT, PDF, DOCX)
- Enter one or more URLs to scrape (default: DAAD scholarships)
- Scrape and extract opportunities
- Embed user profile and opportunities using MiniLM
- Retrieve the most relevant matches using FAISS
- Display results in a simple UI

## Setup
- Python 3.8+
- Install dependencies:
  ```bash
  pip install gradio faiss-cpu sentence-transformers firecrawl requests pdfplumber python-docx
  ```
- Set your Firecrawl API key as an environment variable:
  ```bash
  export FIRECRAWL_API_KEY=your_actual_api_key_here
  ```

## Usage
- Run the app:
  ```bash
  python app.py
  ```
- Open the provided URL in your browser.

---

This README describes the HuggingFace Space tool for matching user profiles to scraped opportunities. See `app.py` for implementation details.

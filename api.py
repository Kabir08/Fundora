"""
FastAPI backend for the Scholarship Matcher ChatGPT Action.
Scrapes preset scholarship sites, stores embeddings in a persistent ChromaDB
vector database (on disk), and serves a /match endpoint that ChatGPT calls
with a user's profile text.  Scholarships are re-scraped only every 7 days;
between restarts the index is read from disk — keeping RAM usage low on Render.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import urljoin, urlparse

import numpy as np
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from playwright.sync_api import sync_playwright
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "scholarships"
META_FILE = "./chroma_db/meta.json"
SCRAPE_TTL_DAYS = 7          # Re-scrape only if index is older than this
ENCODE_BATCH_SIZE = 64       # Encode in batches to cap peak RAM
MAX_PAGES_PER_SITE = 4
PAGE_DELAY_SECONDS = 1.0
DEFAULT_TOP_K = 10

# (name, url_template, paginated)
# For paginated sites use {page} placeholder; page numbering starts at 1.
PRESET_SITES: list[tuple[str, str, bool]] = [
    # ── Global aggregators ────────────────────────────────────────────────
    ("OpportunityDesk",      "https://opportunitydesk.org/category/scholarships/page/{page}/", True),
    ("Scholars4Dev",         "https://www.scholars4dev.com/category/scholarships/page/{page}/", True),
    ("ScholarshipPortal",    "https://www.scholarshipportal.com/scholarships/?page={page}", True),
    ("FindAPhD",             "https://www.findaphd.com/phds/", False),
    ("ScholarshipsForDev",   "https://scholarshipsfordevelopment.org/scholarships/page/{page}/", True),
    ("InternationalScholarships", "https://www.internationalscholarships.com/", False),
    ("CareerFoundry",        "https://careerfoundry.com/en/blog/career-change/scholarships/", False),

    # ── Europe ────────────────────────────────────────────────────────────
    ("DAAD",                 "https://www2.daad.de/deutschland/stipendium/datenbank/en/21148-scholarship-database/?status=3&page={page}", True),
    ("EURAXESS",             "https://euraxess.ec.europa.eu/jobs/search", False),
    ("ErasmusPlus",          "https://erasmus-plus.ec.europa.eu/opportunities/opportunities-for-individuals/students", False),
    ("HeinrichBoell",        "https://www.boell.de/en/stipendien", False),
    ("Chevening",            "https://www.chevening.org/scholarships/", False),
    ("Commonwealth",         "https://cscuk.fcdo.gov.uk/scholarships/", False),
    ("UCL",                  "https://www.ucl.ac.uk/scholarships/scholarships-students-outside-uk", False),
    ("GatesOxford",          "https://www.ox.ac.uk/admissions/graduate/fees-and-funding/fees-funding-and-scholarship-search/scholarships-1", False),
    ("GatesCambridge",       "https://www.gatescambridge.org/apply/", False),
    ("RhodesScholarship",    "https://www.rhodeshouse.ox.ac.uk/scholarships/the-rhodes-scholarship/", False),
    ("ScholarshipHub",       "https://www.thescholarshiphub.org.uk/scholarships/page/{page}/", True),

    # ── United States ─────────────────────────────────────────────────────
    ("Fulbright",            "https://foreign.fulbrightonline.org/about/foreign-fulbright", False),
    ("Fastweb",              "https://www.fastweb.com/college-scholarships", False),
    ("CollegeBoard",         "https://bigfuture.collegeboard.org/pay-for-college/scholarship-search", False),
    ("GoingMerry",           "https://www.goingmerry.com/resources/scholarships/", False),
    ("Niche",                "https://www.niche.com/colleges/scholarships/", False),
    ("BoldOrg",              "https://bold.org/scholarships/", False),

    # ── Canada ───────────────────────────────────────────────────────────
    ("EduCanada",            "https://www.educanada.ca/scholarships-bourses/index.aspx?lang=eng", False),
    ("VanierScholarship",    "https://vanier.gc.ca/en/home-accueil.html", False),
    ("TrudeauFoundation",    "https://www.trudeaufoundation.ca/programs/phd-scholarships", False),
    ("StellarScholarships",  "https://www.scholarshipscanada.com/Scholarships/FeaturedScholarships.aspx", False),

    # ── Australia ─────────────────────────────────────────────────────────
    ("AustraliaAwards",      "https://www.australiaawards.gov.au/scholarships", False),
    ("StudyInAustralia",     "https://www.studyinaustralia.gov.au/english/australian-scholarships", False),
    ("ANUScholarships",      "https://www.anu.edu.au/study/scholarships/find-a-scholarship", False),
    ("MelbourneUni",         "https://scholarships.unimelb.edu.au/international/find-scholarships", False),

    # ── Asia ─────────────────────────────────────────────────────────────
    ("MEXT-Japan",           "https://www.mext.go.jp/en/policy/education/highered/title02/detail02/sdetail02/1373897.htm", False),
    ("JASSO-Japan",          "https://www.jasso.or.jp/en/study_j/scholarship/", False),
    ("GKS-Korea",            "https://www.studyinkorea.go.kr/en/sub/gks/allnew_invite.do", False),
    ("CSC-China",            "https://www.campuschina.org/scholarships/index.html", False),
    ("SingaporeGovt",        "https://www.moe.gov.sg/financial-matters/scholarships", False),
    ("ASEAN-Scholarships",   "https://www.moe.gov.sg/financial-matters/scholarships/asean", False),
    ("GyanDhan",             "https://www.gyandhan.com/scholarships?page={page}", True),

    # ── Africa / Middle East ──────────────────────────────────────────────
    ("AfterSchoolAfrica",    "https://afterschoolafrica.com/scholarships/page/{page}/", True),
    ("AfricanUnion",         "https://au.int/en/scholarships", False),
    ("MasterCard-Foundation","https://mastercardfoundation.org/programs/scholars-program", False),

    # ── International organisations ───────────────────────────────────────
    ("WorldBankYPP",         "https://www.worldbank.org/en/programs/scholarships", False),
    ("ADBScholarship",       "https://www.adb.org/work-with-us/careers/scholarships", False),
    ("AgaKhan",              "https://www.akdn.org/our-agencies/aga-khan-foundation/international-scholarship-programme", False),
    ("UNScholarships",       "https://www.un.org/en/academic-impact/page/scholarship-opportunities", False),
    ("RotaryFoundation",     "https://www.rotary.org/en/our-programs/scholarships", False),
]

# ---------------------------------------------------------------------------
# Globals (populated by background thread)
# ---------------------------------------------------------------------------

_model: Optional[SentenceTransformer] = None
_chroma_client = None          # chromadb.PersistentClient
_chroma_collection = None      # chromadb.Collection — embeddings live on disk, not RAM
_index_ready = threading.Event()
_index_lock = threading.Lock()
_index_error: Optional[str] = None
_build_started_at: Optional[float] = None

# ---------------------------------------------------------------------------
# Scraping utilities (shared with app.py)
# ---------------------------------------------------------------------------

def _get_base(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def _abs(href: str, page_url: str) -> str:
    return urljoin(page_url, href)


def _collect_sibling_content(heading_tag) -> tuple[str, str]:
    level = int(heading_tag.name[1])
    stop_tags = {f"h{i}" for i in range(1, level + 1)}
    parts, link = [], ""
    node = heading_tag.next_sibling
    while node:
        name = getattr(node, "name", None)
        if name in stop_tags:
            break
        if name:
            text = node.get_text(" ", strip=True)
            if text:
                parts.append(text)
            if not link:
                a = node.find("a", href=True) if hasattr(node, "find") else None
                if a and len(a.get_text(strip=True)) > 2:
                    link = a["href"]
        node = node.next_sibling
    return " ".join(parts)[:600], link


def _fetch_html(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        pg = browser.new_page()
        pg.goto(url, wait_until="networkidle", timeout=30000)
        html = pg.content()
        browser.close()
    return html


def scrape_items(url: str) -> list[dict]:
    try:
        html = _fetch_html(url)
    except Exception as e:
        return [{"title": url, "link": url, "description": f"[Error: {e}]"}]

    soup = BeautifulSoup(html, "html.parser")
    base = _get_base(url)

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    items: list[dict] = []

    # ── Strategy 1: Repeated card containers ──────────────────────────────
    candidate_classes: Counter = Counter()
    for el in soup.find_all(["article", "li", "div"], class_=True):
        for cls in el.get("class", []):
            if any(skip in cls.lower() for skip in [
                "footer", "nav", "menu", "modal", "cookie", "banner",
                "wrapper", "container", "row", "col", "icon", "clearfix",
                "active", "hidden", "visible", "block", "item", "list",
            ]):
                continue
            candidate_classes[cls] += 1

    card_classes = [cls for cls, count in candidate_classes.most_common(5) if count >= 3]
    for cls in card_classes:
        cards = soup.find_all(["article", "li", "div"], class_=cls)
        if len(cards) < 3:
            continue
        batch = []
        for card in cards:
            heading = card.find(["h1", "h2", "h3", "h4", "h5"])
            if not heading:
                continue
            title = heading.get_text(" ", strip=True).strip()
            if len(title) < 5:
                continue
            a = heading.find("a", href=True) or card.find("a", href=True)
            link = _abs(a["href"], url) if a else url
            desc = card.get_text(" ", strip=True)
            batch.append({"title": title, "link": link, "description": desc})
        if len(batch) >= 3:
            items = batch
            break

    # ── Strategy 2: Heading-per-item ──────────────────────────────────────
    if not items:
        main = (soup.find("main")
                or soup.find("div", id=lambda x: x and "content" in x.lower())
                or soup.body)
        for heading_tag in ["h3", "h2", "h4"]:
            headings = main.find_all(heading_tag) if main else []
            if len(headings) < 3:
                continue
            batch = []
            for h in headings:
                title = h.get_text(" ", strip=True).strip()
                if len(title) < 5:
                    continue
                if any(w in title.lower() for w in [
                    "information for", "quick links", "contact us",
                    "follow us", "social media",
                ]):
                    continue
                desc, sibling_link = _collect_sibling_content(h)
                h_id = h.get("id") or (h.find("a") and h.find("a").get("id"))
                if h_id:
                    link = f"{url.split('#')[0]}#{h_id}"
                elif sibling_link:
                    link = _abs(sibling_link, url)
                else:
                    a = h.find("a", href=True)
                    link = _abs(a["href"], url) if a else url
                batch.append({"title": title, "link": link,
                              "description": f"{title}. {desc}"})
            if len(batch) >= 3:
                items = batch
                break

    # ── Strategy 3: Named anchor links ───────────────────────────────────
    if not items:
        seen: set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            title = a.get_text(" ", strip=True)
            if len(title) < 8 or href in seen:
                continue
            seen.add(href)
            link = _abs(href, url)
            parent = a.find_parent(["li", "p", "td", "div"])
            desc = parent.get_text(" ", strip=True) if parent else title
            items.append({"title": title, "link": link, "description": desc})

    # ── Fallback: text chunks ─────────────────────────────────────────────
    if not items:
        text = soup.get_text(" ", strip=True)
        words = text.split()
        for i in range(0, len(words), 450):
            chunk = " ".join(words[i: i + 500])
            items.append({"title": f"Section {i // 450 + 1}", "link": url,
                          "description": chunk})

    return items


def scrape_site(name: str, url_template: str, paginated: bool) -> list[dict]:
    all_items: list[dict] = []
    pages = range(1, MAX_PAGES_PER_SITE + 1) if paginated else [None]
    for page in pages:
        url = url_template.replace("{page}", str(page)) if page else url_template
        batch = scrape_items(url)
        # Tag each item with source
        for it in batch:
            it["source"] = name
        all_items.extend(batch)
        if page:
            time.sleep(PAGE_DELAY_SECONDS)
    return all_items

# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def _init_chroma():
    """Create (or open) the persistent ChromaDB collection."""
    global _chroma_client, _chroma_collection
    import chromadb
    os.makedirs(CHROMA_DIR, exist_ok=True)
    _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    _chroma_collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _collection_is_fresh() -> bool:
    """True if the collection has data scraped within SCRAPE_TTL_DAYS."""
    if _chroma_collection is None or _chroma_collection.count() == 0:
        return False
    if not os.path.exists(META_FILE):
        return False
    with open(META_FILE) as f:
        data = json.load(f)
    age_days = (time.time() - data.get("last_scraped", 0)) / 86400
    return age_days < SCRAPE_TTL_DAYS


def _rebuild_collection():
    """Drop and recreate the ChromaDB collection, returning the new instance."""
    global _chroma_client, _chroma_collection
    import chromadb
    os.makedirs(CHROMA_DIR, exist_ok=True)
    _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        _chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _chroma_collection = _chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _store_to_chroma(items: list[dict]):
    """
    Encode items in small batches (caps peak RAM) and upsert into ChromaDB.
    Embeddings are persisted to disk — not held in memory after this call.
    """
    global _model
    total = len(items)
    for batch_start in range(0, total, ENCODE_BATCH_SIZE):
        batch = items[batch_start: batch_start + ENCODE_BATCH_SIZE]
        texts = [it["description"] for it in batch]
        embs = _model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        ).tolist()
        _chroma_collection.add(
            ids=[f"item_{batch_start + j}" for j in range(len(batch))],
            embeddings=embs,
            documents=texts,
            metadatas=[
                {
                    "title": it["title"][:500],
                    "link": it["link"][:500],
                    "source": it.get("source", ""),
                }
                for it in batch
            ],
        )
    # Write scrape timestamp to meta file
    with open(META_FILE, "w") as f:
        json.dump({"last_scraped": time.time(), "count": total}, f)
    print(f"[index] Stored {total} items to ChromaDB.")


# ---------------------------------------------------------------------------
# Index build / cache
# ---------------------------------------------------------------------------

def _scrape_all() -> list[dict]:
    """Scrape every preset site and return the combined item list."""
    all_items: list[dict] = []
    for name, url_tpl, paginated in PRESET_SITES:
        print(f"[index] Scraping {name}…")
        try:
            batch = scrape_site(name, url_tpl, paginated)
            all_items.extend(batch)
        except Exception as exc:
            print(f"[index] {name} failed: {exc}")
    return all_items


def load_or_build():
    global _model, _index_error, _build_started_at
    _build_started_at = time.time()
    try:
        _init_chroma()

        if _collection_is_fresh():
            count = _chroma_collection.count()
            print(f"[index] ChromaDB is fresh ({count} items). Skipping scrape.")
            if _model is None:
                _model = SentenceTransformer("all-MiniLM-L6-v2")
            _index_ready.set()
            return

        if _model is None:
            _model = SentenceTransformer("all-MiniLM-L6-v2")

        print("[index] Scraping scholarship sites…")
        items = _scrape_all()

        with _index_lock:
            _rebuild_collection()
            _store_to_chroma(items)

        _index_ready.set()
        print(f"[index] Ready — {len(items)} items indexed.")
    except Exception as exc:
        _index_error = str(exc)
        print(f"[index] Build failed: {exc}")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Server is bound and ready — now start background work
    loop = asyncio.get_event_loop()
    t = threading.Thread(target=load_or_build, daemon=True)
    t.start()
    yield


app = FastAPI(
    title="Scholarship Matcher",
    description="Semantic scholarship search for ChatGPT Actions",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ChatGPT Actions require permissive CORS
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class MatchRequest(BaseModel):
    profile: str
    top_k: int = DEFAULT_TOP_K


class ScholarshipResult(BaseModel):
    title: str
    link: str
    description: str
    source: Optional[str] = None


class MatchResponse(BaseModel):
    results: list[ScholarshipResult]
    total_indexed: int
    index_ready: bool


class IndexStatus(BaseModel):
    ready: bool
    total_items: int
    error: Optional[str]
    build_started_at: Optional[float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "index_ready": _index_ready.is_set()}


@app.get("/index_status", response_model=IndexStatus)
def index_status():
    count = _chroma_collection.count() if _chroma_collection is not None else 0
    return IndexStatus(
        ready=_index_ready.is_set(),
        total_items=count,
        error=_index_error,
        build_started_at=_build_started_at,
    )


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    if not _index_ready.is_set():
        raise HTTPException(
            status_code=503,
            detail="Index is still building. Try again in a few minutes.",
        )
    if not req.profile.strip():
        raise HTTPException(status_code=400, detail="profile must not be empty.")

    top_k = max(1, min(req.top_k, 50))

    # Encode query — only the single query vector lives in RAM
    user_emb = _model.encode([req.profile], convert_to_numpy=True).tolist()

    with _index_lock:
        raw = _chroma_collection.query(
            query_embeddings=user_emb,
            n_results=min(top_k * 3, _chroma_collection.count()),
        )

    seen_links: set[str] = set()
    results: list[ScholarshipResult] = []
    for i, _id in enumerate(raw["ids"][0]):
        meta = raw["metadatas"][0][i]
        doc = raw["documents"][0][i]
        link = meta.get("link", "")
        if link in seen_links:
            continue
        seen_links.add(link)
        results.append(ScholarshipResult(
            title=meta.get("title", ""),
            link=link,
            description=doc[:400],
            source=meta.get("source"),
        ))
        if len(results) >= top_k:
            break

    return MatchResponse(
        results=results,
        total_indexed=_chroma_collection.count(),
        index_ready=True,
    )


@app.post("/refresh_index")
def refresh_index():
    """Force rebuild the index (clears ChromaDB collection and re-scrapes)."""
    global _index_error
    _index_error = None
    _index_ready.clear()
    # Remove meta file so freshness check fails and a full rescrape is triggered
    if os.path.exists(META_FILE):
        os.remove(META_FILE)
    t = threading.Thread(target=load_or_build, daemon=True)
    t.start()
    return {"status": "rebuild started"}


@app.get("/privacy", response_class=HTMLResponse)
def privacy_policy():
    """Privacy policy page — required by ChatGPT Actions."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Privacy Policy — Fundora Scholarship Matcher</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 20px; line-height: 1.7; color: #222; }
    h1 { font-size: 1.8rem; margin-bottom: 4px; }
    h2 { font-size: 1.2rem; margin-top: 2rem; }
    p, li { font-size: 0.97rem; }
    a { color: #0066cc; }
    footer { margin-top: 3rem; font-size: 0.85rem; color: #666; }
  </style>
</head>
<body>
  <h1>Privacy Policy</h1>
  <p><strong>Fundora Scholarship Matcher</strong> &mdash; Last updated: April 20, 2026</p>

  <h2>1. Overview</h2>
  <p>
    Fundora Scholarship Matcher (&ldquo;the Service&rdquo;) is an AI-powered tool that helps students
    find relevant scholarship opportunities based on a free-text academic profile they provide.
    This Privacy Policy explains what information we collect, how it is used, and your rights.
  </p>

  <h2>2. Information We Collect</h2>
  <p>We collect only the information you voluntarily submit when using the Service:</p>
  <ul>
    <li><strong>Profile text</strong> — the academic background description you type or paste into the
    search field (e.g. degree level, field of study, nationality).</li>
    <li><strong>Usage metadata</strong> — standard web-server logs such as IP address, timestamp, and
    HTTP method, retained for up to 30 days for security and debugging purposes.</li>
  </ul>
  <p>We do <strong>not</strong> collect names, email addresses, payment information, or any account
  credentials.</p>

  <h2>3. How We Use Your Information</h2>
  <ul>
    <li>To perform a semantic similarity search against our scholarship index and return relevant results.</li>
    <li>To monitor service health and fix errors.</li>
  </ul>
  <p>We do <strong>not</strong> sell, rent, or share your profile text with third parties.
  Profile text is never stored persistently &mdash; it exists only in memory during the duration of a
  single API request and is discarded immediately after the response is sent.</p>

  <h2>4. Third-Party Data Sources</h2>
  <p>
    The Service scrapes publicly available scholarship listings from third-party websites
    spanning multiple regions — including DAAD, Chevening, Commonwealth, Fulbright, Erasmus+,
    Australia Awards, MEXT, GKS, CSC, Vanier, World Bank, and many others. We do not control
    the privacy practices of those sites. The scraped content is stored in a local vector
    database and refreshed automatically every 7 days to reduce load on external servers.
  </p>

  <h2>5. ChatGPT / OpenAI Integration</h2>
  <p>
    When accessed through a ChatGPT Custom GPT, your profile text is transmitted from
    OpenAI&rsquo;s servers to this API over HTTPS. OpenAI&rsquo;s own
    <a href="https://openai.com/policies/privacy-policy" target="_blank" rel="noopener">Privacy Policy</a>
    governs how ChatGPT handles your conversations.
  </p>

  <h2>6. Data Security</h2>
  <p>
    All data in transit is protected by TLS (HTTPS). The Service is hosted on Render.com;
    Render&rsquo;s infrastructure security practices apply to data at rest.
  </p>

  <h2>7. Children&rsquo;s Privacy</h2>
  <p>
    The Service is not directed at children under 13. We do not knowingly collect information
    from children under 13.
  </p>

  <h2>8. Changes to This Policy</h2>
  <p>
    We may update this policy from time to time. The &ldquo;Last updated&rdquo; date at the top
    of this page will reflect any changes.
  </p>

  <h2>9. Contact</h2>
  <p>
    If you have questions about this Privacy Policy, please open an issue on our
    <a href="https://github.com/Kabir08/Fundora" target="_blank" rel="noopener">GitHub repository</a>.
  </p>

  <footer>Fundora Scholarship Matcher &mdash; <a href="/">API Docs</a></footer>
</body>
</html>
"""

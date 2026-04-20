import io
import gradio as gr
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer
import faiss

# ---------------------------------------------------------------------------
# Model (loaded once at startup)
# ---------------------------------------------------------------------------
print("Loading MiniLM model…")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model ready.")



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text_from_file(file_path: str) -> str:
    """Extract plain text from a TXT, PDF, or DOCX file."""
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        with open(file_path, "r", errors="ignore") as f:
            return f.read()


def _get_base(url: str) -> str:
    from urllib.parse import urlparse
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def _abs(href: str, base: str, page_url: str) -> str:
    from urllib.parse import urljoin
    if href.startswith("//"):
        from urllib.parse import urlparse
        scheme = urlparse(page_url).scheme
        return f"{scheme}:{href}"
    return urljoin(page_url, href)


def _collect_sibling_content(heading_tag) -> tuple[str, str]:
    """Walk siblings of a heading tag until the next heading of same/higher level.
    Returns (description_text, first_external_link_href)."""
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


def scrape_items(url: str) -> list[dict]:
    """
    Generic scraper that tries multiple strategies to extract individual items
    (scholarships, opportunities, programmes) from any page. Returns a list of
    dicts with keys: title, link, description.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            pg = browser.new_page()
            pg.goto(url, wait_until="networkidle", timeout=30000)
            html = pg.content()
            browser.close()
    except Exception as e:
        return [{"title": url, "link": url, "description": f"[Error: {e}]"}]

    soup = BeautifulSoup(html, "html.parser")
    base = _get_base(url)

    # Strip global noise
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    items = []

    # ── Strategy 1: Repeated card containers (article, li, div with consistent class) ──
    from collections import Counter
    candidate_classes: Counter = Counter()
    for el in soup.find_all(["article", "li", "div"], class_=True):
        for cls in el.get("class", []):
            # Skip utility/layout/generic classes
            if any(skip in cls.lower() for skip in [
                "footer", "nav", "menu", "modal", "cookie", "banner",
                "wrapper", "container", "row", "col", "icon", "clearfix",
                "active", "hidden", "visible", "block", "item", "list",
            ]):
                continue
            candidate_classes[cls] += 1

    # Pick classes that repeat ≥3 times (list of cards)
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
            # Find best link in card
            a = heading.find("a", href=True) or card.find("a", href=True)
            link = _abs(a["href"], base, url) if a else url
            desc = card.get_text(" ", strip=True)
            batch.append({"title": title, "link": link, "description": desc})
        if len(batch) >= 3:
            items = batch
            break

    # ── Strategy 2: Heading-per-item pattern (h2/h3 each = one scholarship/programme) ──
    if not items:
        main = soup.find("main") or soup.find("div", id=lambda x: x and "content" in x.lower()) or soup.body
        for heading_tag in ["h3", "h2", "h4"]:
            headings = main.find_all(heading_tag) if main else []
            if len(headings) < 3:
                continue
            batch = []
            for h in headings:
                title = h.get_text(" ", strip=True).strip()
                if len(title) < 5:
                    continue
                # Skip nav/footer headings
                if any(word in title.lower() for word in ["information for", "quick links", "contact us", "follow us", "about ucl", "social media"]):
                    continue
                desc, sibling_link = _collect_sibling_content(h)
                # Prefer anchor on the heading itself, then sibling link
                h_id = h.get("id") or (h.find("a") and h.find("a").get("id"))
                if h_id:
                    link = f"{url.split('#')[0]}#{h_id}"
                elif sibling_link:
                    link = _abs(sibling_link, base, url)
                else:
                    a = h.find("a", href=True)
                    link = _abs(a["href"], base, url) if a else url
                batch.append({"title": title, "link": link, "description": f"{title}. {desc}"})
            if len(batch) >= 3:
                items = batch
                break

    # ── Strategy 3: Named anchor links in a table-of-contents section ──
    if not items:
        seen = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            title = a.get_text(" ", strip=True)
            if len(title) < 8 or href in seen:
                continue
            seen.add(href)
            link = _abs(href, base, url)
            parent = a.find_parent(["li", "p", "td", "div"])
            desc = parent.get_text(" ", strip=True) if parent else title
            items.append({"title": title, "link": link, "description": desc})

    # ── Fallback: overlapping text chunks of entire page ──
    if not items:
        text = soup.get_text(" ", strip=True)
        words = text.split()
        for i in range(0, len(words), 450):
            chunk = " ".join(words[i : i + 500])
            items.append({"title": f"Section {i//450+1}", "link": url, "description": chunk})

    return items


def retrieve_top_items(user_text: str, items: list[dict], top_k: int = 5) -> list[dict]:
    """Embed user text and item descriptions, use FAISS to find best matches."""
    if not items:
        return []
    texts = [it["description"] for it in items]
    user_emb = model.encode([user_text], convert_to_numpy=True).astype("float32")
    item_embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatL2(item_embs.shape[1])
    index.add(item_embs)
    distances, indices = index.search(user_emb, min(top_k, len(items)))
    # Deduplicate by link
    seen_links = set()
    results = []
    for i in indices[0]:
        it = items[i]
        if it["link"] not in seen_links:
            seen_links.add(it["link"])
            results.append(it)
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(profile_file, urls_input: str, top_k: int = 5):
    if profile_file is None:
        return "Please upload a profile file."
    if not urls_input.strip():
        return "Please enter at least one URL."

    # 1. Extract user profile text
    user_text = extract_text_from_file(profile_file)
    if not user_text.strip():
        return "Could not extract text from your profile file."

    # 2. Scrape each URL and extract individual items
    urls = [u.strip() for u in urls_input.split(',') if u.strip()]
    all_items = []
    scrape_log = []
    for url in urls:
        items = scrape_items(url)
        all_items.extend(items)
        scrape_log.append(f"✓ {url} → {len(items)} items found")

    if not all_items:
        return "No content could be scraped from the provided URLs."

    # 3. Retrieve top matching items
    top_items = retrieve_top_items(user_text, all_items, top_k=top_k)

    # 4. Format output — show title + link only (clean list)
    output_lines = ["## Top Matching Opportunities\n", "\n".join(scrape_log), "\n---\n"]
    for i, item in enumerate(top_items, 1):
        output_lines.append(f"{i}. [{item['title']}]({item['link']})\n")
    return "\n".join(output_lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DEFAULT_URL = (
    "https://www2.daad.de/deutschland/stipendium/datenbank/en/21148-scholarship-database/"
    "?status=3&origin=&subjectGrps=F&daad=&intention=&q=&page=1&back=1"
)

with gr.Blocks(title="Opportunity Matcher") as demo:
    gr.Markdown(
        "# Scholarship & Opportunity Matcher\n"
        "Upload your profile/background file and paste any URLs (comma-separated). "
        "The tool scrapes the pages, encodes all content with MiniLM, and retrieves "
        "the most relevant sections for your profile using FAISS."
    )
    with gr.Row():
        with gr.Column():
            profile_file = gr.File(
                label="Your Profile / Background (TXT, PDF, DOCX)",
                file_types=[".txt", ".pdf", ".docx"]
            )
            urls_input = gr.Textbox(
                label="URLs to scrape (comma-separated)",
                value=DEFAULT_URL,
                lines=3
            )
            top_k = gr.Slider(1, 20, value=5, step=1, label="Number of top matches")
            btn = gr.Button("Find Opportunities", variant="primary")
        with gr.Column():
            output = gr.Markdown(label="Results")

    btn.click(run_pipeline, inputs=[profile_file, urls_input, top_k], outputs=output)

if __name__ == "__main__":
    demo.launch()


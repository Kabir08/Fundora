import io
from collections import Counter
from urllib.parse import urljoin, urlparse

import gradio as gr
import httpx
import pdfplumber
from bs4 import BeautifulSoup
from docx import Document

# ---------------------------------------------------------------------------
# Model — imported from api.py so we share the single loaded instance.
# api.py populates _model lazily in its background thread; app.py references
# it through the module so it always sees the latest value.
# ---------------------------------------------------------------------------
import api as _api_module

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}



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
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def _abs(href: str, base: str, page_url: str) -> str:
    if href.startswith("//"):
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
    Generic scraper using httpx (no browser required).
    Tries multiple strategies to extract individual items from any page.
    """
    try:
        with httpx.Client(headers=_HTTP_HEADERS, follow_redirects=True, timeout=20.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
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


import numpy as np
import faiss


def retrieve_top_items(user_text: str, items: list[dict], top_k: int = 5) -> list[dict]:
    """Embed user text and items using the shared model, rank with FAISS."""
    if not items:
        return []
    model = _api_module._model
    if model is None:
        return []
    texts = [it["description"] for it in items]
    user_emb = model.encode([user_text], convert_to_numpy=True).astype("float32")
    item_embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    idx = faiss.IndexFlatL2(item_embs.shape[1])
    idx.add(item_embs)
    _, indices = idx.search(user_emb, min(top_k, len(items)))
    seen_links: set[str] = set()
    results = []
    for i in indices[0]:
        it = items[i]
        if it["link"] not in seen_links:
            seen_links.add(it["link"])
            results.append(it)
    return results


# ---------------------------------------------------------------------------
# Tab 1: Quick Match — uses the pre-loaded seed index via /match endpoint
# ---------------------------------------------------------------------------

def _extract_profile_text(profile_file) -> str:
    if profile_file is None:
        return ""
    return extract_text_from_file(profile_file)


def quick_match(profile_file, profile_text: str, top_k: int) -> str:
    """Match against the pre-loaded seed index (instant — no scraping)."""
    text = profile_text.strip()
    if not text and profile_file is not None:
        text = _extract_profile_text(profile_file)
    if not text:
        return "⚠️ Please upload a resume/CV file or type your profile in the text box."

    if not _api_module._index_ready.is_set():
        return "⏳ The scholarship index is still loading. This usually takes 10–30 seconds. Please try again shortly."

    # Call the match logic directly (in-process, no HTTP round-trip)
    from api import MatchRequest, match as _match
    try:
        resp = _match(MatchRequest(profile=text, top_k=int(top_k)))
    except Exception as e:
        return f"❌ Error: {e}"

    if not resp.results:
        return "No matching scholarships found. Try broadening your profile description."

    lines = [f"### 🎓 Top {len(resp.results)} Scholarships for Your Profile\n",
             f"*Searched {resp.total_indexed} indexed scholarships*\n\n---\n"]
    for i, r in enumerate(resp.results, 1):
        lines.append(f"**{i}. [{r.title}]({r.link})**")
        if r.source:
            lines.append(f"  *Source: {r.source}*")
        lines.append(f"  {r.description[:250]}…\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 2: Custom Search — user provides URLs, scrapes on-the-fly with httpx
# ---------------------------------------------------------------------------

def custom_search(profile_file, profile_text: str, urls_input: str, top_k: int) -> str:
    text = profile_text.strip()
    if not text and profile_file is not None:
        text = _extract_profile_text(profile_file)
    if not text:
        return "⚠️ Please upload a resume/CV file or type your profile in the text box."
    if not urls_input.strip():
        return "⚠️ Please enter at least one URL to search."

    urls = [u.strip() for u in urls_input.split(",") if u.strip()]
    all_items: list[dict] = []
    log_lines: list[str] = []
    for url in urls:
        batch = scrape_items(url)
        valid = [b for b in batch if not b["description"].startswith("[Error")]
        all_items.extend(valid)
        log_lines.append(f"- `{url}` → {len(valid)} items")

    if not all_items:
        return "❌ No content could be scraped from the provided URLs. They may block bots or require JavaScript."

    top_items = retrieve_top_items(text, all_items, top_k=int(top_k))
    if not top_items:
        return "No matching items found."

    lines = [f"### 🔍 Top {len(top_items)} Matches from Custom URLs\n",
             "**Scrape log:**\n" + "\n".join(log_lines) + "\n\n---\n"]
    for i, item in enumerate(top_items, 1):
        lines.append(f"**{i}. [{item['title']}]({item['link']})**")
        lines.append(f"  {item['description'][:250]}…\n")
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# Gradio UI — two tabs
# ---------------------------------------------------------------------------

_PROFILE_HELP = "Upload a PDF/DOCX resume **or** type/paste your background below."

with gr.Blocks(
    title="Fundora — Scholarship Matcher",
    theme=gr.themes.Soft(),
    css=".gr-button-primary { background: #2563eb !important; }",
) as demo:
    gr.Markdown(
        "# 🎓 Fundora — AI Scholarship Matcher\n"
        "Find scholarships that match your profile from a curated index of 90+ "
        "global programmes (Chevening, DAAD, Fulbright, Google PhD, MEXT, Vanier, and more).\n\n"
        "> **Free to use. No login required.**"
    )

    with gr.Tabs():
        # ── Tab 1: Quick Match (seed index) ───────────────────────────────
        with gr.TabItem("⚡ Quick Match  (recommended)"):
            gr.Markdown(
                "Match against our **pre-loaded index of 90+ scholarships** — results in seconds."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    qm_file = gr.File(
                        label="Upload Resume / CV  (PDF, DOCX, TXT)",
                        file_types=[".txt", ".pdf", ".docx"],
                    )
                    qm_text = gr.Textbox(
                        label="Or paste / type your profile here",
                        placeholder=(
                            "e.g. Indian student, BE Electronics, GPA 8.0, "
                            "ML research, LLM quantization, seeking MS abroad…"
                        ),
                        lines=5,
                    )
                    qm_topk = gr.Slider(1, 20, value=8, step=1, label="Results to show")
                    qm_btn = gr.Button("Find Scholarships", variant="primary")
                with gr.Column(scale=2):
                    qm_out = gr.Markdown(label="Results")

            qm_btn.click(
                quick_match,
                inputs=[qm_file, qm_text, qm_topk],
                outputs=qm_out,
            )

        # ── Tab 2: Custom Search (user-provided URLs) ──────────────────────
        with gr.TabItem("🔍 Custom Search  (any URL)"):
            gr.Markdown(
                "Paste **any scholarship or opportunity page URLs** — Fundora will scrape "
                "and rank them against your profile on the fly."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    cs_file = gr.File(
                        label="Upload Resume / CV  (PDF, DOCX, TXT)",
                        file_types=[".txt", ".pdf", ".docx"],
                    )
                    cs_text = gr.Textbox(
                        label="Or paste / type your profile here",
                        placeholder="e.g. Nigerian student, BSc Computer Science, 3.8 GPA, data science focus…",
                        lines=5,
                    )
                    cs_urls = gr.Textbox(
                        label="URLs to search (comma-separated)",
                        placeholder="https://www.daad.de/..., https://opportunitydesk.org/...",
                        lines=3,
                    )
                    cs_topk = gr.Slider(1, 20, value=5, step=1, label="Results to show")
                    cs_btn = gr.Button("Search", variant="primary")
                with gr.Column(scale=2):
                    cs_out = gr.Markdown(label="Results")

            cs_btn.click(
                custom_search,
                inputs=[cs_file, cs_text, cs_urls, cs_topk],
                outputs=cs_out,
            )

    gr.Markdown(
        "---\n"
        "Made with ❤️ by [Kabir Potdar](https://github.com/Kabir08) · "
        "[GitHub](https://github.com/Kabir08/Fundora) · "
        "[API docs](/docs) · [Privacy](/privacy)"
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)



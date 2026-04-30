"""
generate_seed.py — Run this locally to create scholarships_seed.json.

Scrapes a curated subset of scholarship sites using httpx (no browser needed).
The resulting JSON is committed to the repo so Render cold-starts load it
in ~10 seconds instead of spending 30-60 minutes scraping.

Usage:
    python generate_seed.py

Output:
    scholarships_seed.json  (commit this file)
"""
from __future__ import annotations

import json
import time

import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import Counter

# ── Curated list: reliable static-HTML sites only ─────────────────────────
# These all render full scholarship listings without JavaScript.
SEED_SITES: list[tuple[str, str, bool]] = [
    ("OpportunityDesk",      "https://opportunitydesk.org/category/scholarships/page/{page}/", True),
    ("Scholars4Dev",         "https://www.scholars4dev.com/category/scholarships/page/{page}/", True),
    ("ScholarshipsForDev",   "https://scholarshipsfordevelopment.org/scholarships/page/{page}/", True),
    ("AfterSchoolAfrica",    "https://afterschoolafrica.com/scholarships/page/{page}/", True),
    ("ScholarshipHub",       "https://www.thescholarshiphub.org.uk/scholarships/page/{page}/", True),
    ("Chevening",            "https://www.chevening.org/scholarships/", False),
    ("Commonwealth",         "https://cscuk.fcdo.gov.uk/scholarships/", False),
    ("GatesCambridge",       "https://www.gatescambridge.org/apply/", False),
    ("DAAD",                 "https://www2.daad.de/deutschland/stipendium/datenbank/en/21148-scholarship-database/?status=3&page={page}", True),
    ("AustraliaAwards",      "https://www.australiaawards.gov.au/scholarships", False),
    ("Fulbright",            "https://foreign.fulbrightonline.org/about/foreign-fulbright", False),
    ("GyanDhan",             "https://www.gyandhan.com/scholarships?page={page}", True),
    ("WorldBankYPP",         "https://www.worldbank.org/en/programs/scholarships", False),
    ("RotaryFoundation",     "https://www.rotary.org/en/our-programs/scholarships", False),
    ("AgaKhan",              "https://www.akdn.org/our-agencies/aga-khan-foundation/international-scholarship-programme", False),
    ("MasterCard-Foundation","https://mastercardfoundation.org/programs/scholars-program", False),
    ("ErasmusPlus",          "https://erasmus-plus.ec.europa.eu/opportunities/opportunities-for-individuals/students", False),
    ("RhodesScholarship",    "https://www.rhodeshouse.ox.ac.uk/scholarships/the-rhodes-scholarship/", False),
    ("VanierScholarship",    "https://vanier.gc.ca/en/home-accueil.html", False),
    ("MEXT-Japan",           "https://www.mext.go.jp/en/policy/education/highered/title02/detail02/sdetail02/1373897.htm", False),
]

MAX_PAGES_PER_SITE = 3
PAGE_DELAY = 0.5
OUTPUT_FILE = "scholarships_seed.json"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
}


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


def fetch_html(url: str) -> str:
    with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=20.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text


def scrape_items(url: str) -> list[dict]:
    try:
        html = fetch_html(url)
    except Exception as e:
        print(f"  [skip] {url}: {e}")
        return []

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    items: list[dict] = []

    # Strategy 1: Repeated card containers
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

    # Strategy 2: Heading-per-item
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

    # Strategy 3: Named anchor links
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

    return items


def scrape_site(name: str, url_template: str, paginated: bool) -> list[dict]:
    all_items: list[dict] = []
    pages = range(1, MAX_PAGES_PER_SITE + 1) if paginated else [None]
    for page in pages:
        url = url_template.replace("{page}", str(page)) if page else url_template
        print(f"  Fetching: {url}")
        batch = scrape_items(url)
        for it in batch:
            it["source"] = name
        all_items.extend(batch)
        if page:
            time.sleep(PAGE_DELAY)
    return all_items


def main():
    all_items: list[dict] = []
    for name, url_tpl, paginated in SEED_SITES:
        print(f"\n[{name}]")
        try:
            batch = scrape_site(name, url_tpl, paginated)
            print(f"  → {len(batch)} items")
            all_items.extend(batch)
        except Exception as exc:
            print(f"  → FAILED: {exc}")

    # Deduplicate by link
    seen_links: set[str] = set()
    unique: list[dict] = []
    for item in all_items:
        link = item.get("link", "")
        if link not in seen_links:
            seen_links.add(link)
            unique.append(item)

    print(f"\nTotal unique items: {len(unique)}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

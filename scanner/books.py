"""ISBN lookup via Open Library."""

import requests

OPEN_LIBRARY = "https://openlibrary.org"


def lookup(isbn, user_agent):
    """Return a flat dict of book metadata, or None if Open Library has no record."""
    session = requests.Session()
    session.headers["User-Agent"] = user_agent

    resp = session.get(
        f"{OPEN_LIBRARY}/api/books",
        params={"bibkeys": f"ISBN:{isbn}", "format": "json", "jscmd": "data"},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json().get(f"ISBN:{isbn}")
    if not data:
        return None

    def names(key):
        return "; ".join(item["name"] for item in data.get(key, []) if item.get("name"))

    ids = data.get("identifiers", {})
    classes = data.get("classifications", {})
    return {
        "isbn": isbn,
        "title": data.get("title", ""),
        "subtitle": data.get("subtitle", ""),
        "authors": names("authors"),
        "publishers": names("publishers"),
        "publish_date": data.get("publish_date", ""),
        "pages": data.get("number_of_pages", ""),
        "subjects": "; ".join(s["name"] for s in data.get("subjects", [])[:10]),
        "lccn": ", ".join(ids.get("lccn", [])),
        "oclc": ", ".join(ids.get("oclc", [])),
        "lc_classification": ", ".join(classes.get("lc_classifications", [])),
        "dewey": ", ".join(classes.get("dewey_decimal_class", [])),
        "cover_url": data.get("cover", {}).get("large", ""),
        "open_library_url": data.get("url", ""),
    }

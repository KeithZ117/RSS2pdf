#!/usr/bin/env python3
"""Fetch RSS/Atom feeds from OPML and export unseen entries to a PDF (stdlib only)."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import re
import textwrap
import urllib.error
import urllib.request
from html import unescape
from pathlib import Path
from xml.etree import ElementTree as ET

USER_AGENT = "RSS2PDF/1.0 (+local)"
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

PAGE_WIDTH = 595
PAGE_HEIGHT = 842
MARGIN = 36
LINE_HEIGHT = 14
FONT_SIZE = 10
TITLE_WRAP = 90
BODY_WRAP = 95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read OPML feeds and export unseen entries to a PDF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--opml", default="feeds.opml", help="OPML file path")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument(
        "--state-file",
        default=".rss2pdf/state.json",
        help="State file used for deduplication",
    )
    parser.add_argument(
        "--max-per-feed", type=int, default=5, help="Max unseen entries per feed"
    )
    parser.add_argument("--timeout", type=int, default=10, help="HTTP timeout in seconds")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent workers used to fetch feeds",
    )
    parser.add_argument(
        "--summary-chars",
        type=int,
        default=1200,
        help="Max summary length for each entry",
    )
    return parser.parse_args()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def text_from_node(node: ET.Element) -> str:
    return "".join(node.itertext()).strip()


def first_text_by_local(node: ET.Element, names: list[str]) -> str:
    allowed = {name.lower() for name in names}
    for child in node:
        if local_name(child.tag).lower() in allowed:
            text = text_from_node(child)
            if text:
                return text
    return ""


def clean_text(raw: str | None) -> str:
    if not raw:
        return ""
    text = unescape(raw)
    text = TAG_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def parse_opml(opml_path: Path) -> list[dict[str, str]]:
    tree = ET.parse(opml_path)
    root = tree.getroot()
    feeds: list[dict[str, str]] = []
    for node in root.findall(".//outline[@xmlUrl]"):
        url = (node.attrib.get("xmlUrl") or "").strip()
        if not url:
            continue
        title = (node.attrib.get("title") or node.attrib.get("text") or url).strip()
        feeds.append({"title": title, "xmlUrl": url})
    return feeds


def load_state(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    seen = payload.get("seen_keys", [])
    if not isinstance(seen, list):
        return set()
    return {str(item) for item in seen}


def save_state(state_path: Path, seen_keys: set[str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seen_keys": sorted(seen_keys)}
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_entry_key(feed_url: str, entry: dict[str, str]) -> str:
    stable = entry.get("id") or entry.get("guid") or entry.get("link")
    if stable:
        raw = f"{feed_url}|{stable}"
    else:
        raw = "|".join(
            [
                feed_url,
                clean_text(entry.get("title", "")),
                clean_text(entry.get("published", "")),
                clean_text(entry.get("summary", ""))[:200],
            ]
        )
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


def fetch_feed_bytes(url: str, timeout_sec: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/rss+xml, application/atom+xml, application/xml,text/xml,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        return response.read()


def fetch_one_feed(feed: dict[str, str], timeout_sec: int) -> dict[str, object]:
    feed_url = feed["xmlUrl"]
    feed_label = feed["title"]
    try:
        feed_bytes = fetch_feed_bytes(feed_url, timeout_sec)
        parsed_title, parsed_entries = parse_feed(feed_bytes)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        return {"feed": feed, "error": f"[network] {feed_url} -> {exc}"}
    except ET.ParseError as exc:
        return {"feed": feed, "error": f"[xml] {feed_url} -> {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"feed": feed, "error": f"[error] {feed_url} -> {exc}"}

    feed_title = clean_text(parsed_title) or feed_label
    return {
        "feed": feed,
        "feed_title": feed_title,
        "entries": parsed_entries,
        "error": "",
    }


def parse_atom_entries(root: ET.Element) -> tuple[str, list[dict[str, str]]]:
    feed_title = first_text_by_local(root, ["title"])
    entries: list[dict[str, str]] = []

    for child in root:
        if local_name(child.tag).lower() != "entry":
            continue

        title = first_text_by_local(child, ["title"]) or "(Untitled)"
        entry_id = first_text_by_local(child, ["id"])
        published = first_text_by_local(child, ["published", "updated"])
        summary = first_text_by_local(child, ["summary", "content"])

        link = ""
        fallback_link = ""
        for link_node in child:
            if local_name(link_node.tag).lower() != "link":
                continue
            href = (link_node.attrib.get("href") or "").strip()
            if not href:
                continue
            rel = (link_node.attrib.get("rel") or "alternate").strip().lower()
            if not fallback_link:
                fallback_link = href
            if rel in ("", "alternate"):
                link = href
                break
        if not link:
            link = fallback_link

        entries.append(
            {
                "id": entry_id,
                "guid": "",
                "title": title,
                "link": link,
                "published": published,
                "summary": summary,
            }
        )

    return feed_title, entries


def parse_rss_entries(root: ET.Element) -> tuple[str, list[dict[str, str]]]:
    channel: ET.Element | None = None
    for child in root:
        if local_name(child.tag).lower() == "channel":
            channel = child
            break

    container = channel if channel is not None else root
    feed_title = first_text_by_local(container, ["title"])
    entries: list[dict[str, str]] = []

    for node in container.iter():
        if local_name(node.tag).lower() != "item":
            continue

        entries.append(
            {
                "id": "",
                "guid": first_text_by_local(node, ["guid"]),
                "title": first_text_by_local(node, ["title"]) or "(Untitled)",
                "link": first_text_by_local(node, ["link"]),
                "published": first_text_by_local(
                    node, ["pubDate", "published", "updated", "date"]
                ),
                "summary": first_text_by_local(
                    node, ["description", "encoded", "summary", "content"]
                ),
            }
        )

    return feed_title, entries


def parse_feed(feed_bytes: bytes) -> tuple[str, list[dict[str, str]]]:
    root = ET.fromstring(feed_bytes)
    root_kind = local_name(root.tag).lower()
    if root_kind == "feed":
        return parse_atom_entries(root)
    return parse_rss_entries(root)


def collect_new_entries(
    feeds: list[dict[str, str]],
    seen_keys: set[str],
    max_per_feed: int,
    timeout_sec: int,
    summary_chars: int,
    workers: int,
) -> tuple[list[dict[str, str]], list[str], set[str]]:
    collected: list[dict[str, str]] = []
    errors: list[str] = []
    new_seen: set[str] = set()
    total = len(feeds)
    done = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(fetch_one_feed, feed, timeout_sec) for feed in feeds]
        for future in concurrent.futures.as_completed(futures):
            done += 1
            result = future.result()
            feed = result["feed"]
            feed_url = str(feed["xmlUrl"])
            feed_label = str(feed["title"])

            error = str(result["error"])
            if error:
                errors.append(error)
                print(f"[{done}/{total}] fail {feed_label}", flush=True)
                continue

            feed_title = str(result["feed_title"])
            parsed_entries = result["entries"]
            if not isinstance(parsed_entries, list):
                parsed_entries = []

            added = 0
            for raw_entry in parsed_entries:
                if not isinstance(raw_entry, dict):
                    continue

                key = make_entry_key(feed_url, raw_entry)
                if key in seen_keys or key in new_seen:
                    continue
                if added >= max_per_feed:
                    break

                title = clean_text(raw_entry.get("title", "")) or "(Untitled)"
                link = clean_text(raw_entry.get("link", ""))
                published = clean_text(raw_entry.get("published", ""))
                summary = clean_text(raw_entry.get("summary", ""))
                if len(summary) > summary_chars:
                    summary = summary[: summary_chars - 1].rstrip() + "..."

                collected.append(
                    {
                        "key": key,
                        "feed_title": feed_title,
                        "feed_url": feed_url,
                        "title": title,
                        "link": link,
                        "published": published,
                        "summary": summary,
                    }
                )
                new_seen.add(key)
                added += 1

            print(f"[{done}/{total}] ok {feed_title} (+{added})", flush=True)

    return collected, errors, new_seen


def wrap_text(value: str, width: int) -> list[str]:
    if not value:
        return [""]
    return textwrap.wrap(
        value,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )


def build_pdf_lines(entries: list[dict[str, str]], generated_at: dt.datetime) -> list[str]:
    lines = ["RSS Digest", f"Generated at: {generated_at.strftime('%Y-%m-%d %H:%M:%S')}", ""]

    if not entries:
        lines.append("No unseen entries found in this run.")
        return lines

    for idx, entry in enumerate(entries, start=1):
        lines.extend(wrap_text(f"{idx}. {entry['title']}", TITLE_WRAP))
        lines.extend(
            wrap_text(
                f"Source: {entry['feed_title']} | Published: {entry['published'] or '-'}",
                BODY_WRAP,
            )
        )

        if entry["link"]:
            lines.extend(wrap_text(f"Link: {entry['link']}", BODY_WRAP))

        summary = entry["summary"] or "(No summary)"
        lines.extend(wrap_text(summary, BODY_WRAP))
        lines.append("")

    return lines


def escape_pdf_text(value: str) -> str:
    safe = value.encode("latin-1", errors="replace").decode("latin-1")
    safe = safe.replace("\\", "\\\\")
    safe = safe.replace("(", "\\(")
    safe = safe.replace(")", "\\)")
    return safe


def build_page_stream(lines: list[str]) -> bytes:
    if not lines:
        lines = [""]

    commands: list[str] = [
        "BT",
        f"/F1 {FONT_SIZE} Tf",
        f"{LINE_HEIGHT} TL",
        f"{MARGIN} {PAGE_HEIGHT - MARGIN} Td",
    ]

    first = True
    for line in lines:
        if first:
            commands.append(f"({escape_pdf_text(line)}) Tj")
            first = False
        else:
            commands.append("T*")
            commands.append(f"({escape_pdf_text(line)}) Tj")

    commands.append("ET")
    return "\n".join(commands).encode("latin-1", errors="replace")


def paginate_lines(lines: list[str], max_lines: int) -> list[list[str]]:
    if not lines:
        return [[""]]
    pages: list[list[str]] = []
    for i in range(0, len(lines), max_lines):
        pages.append(lines[i : i + max_lines])
    return pages


def write_simple_pdf(pdf_path: Path, pages: list[list[str]]) -> None:
    objects: list[bytes] = []

    def add_obj(content: bytes) -> int:
        objects.append(content)
        return len(objects)

    catalog_id = add_obj(b"")
    pages_id = add_obj(b"")
    font_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids: list[int] = []

    for page in pages:
        stream = build_page_stream(page)
        content_obj = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream"
        )
        content_id = add_obj(content_obj)

        page_obj = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("latin-1")
        page_id = add_obj(page_obj)
        page_ids.append(page_id)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_id - 1] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>"
    ).encode("latin-1")
    objects[catalog_id - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode(
        "latin-1"
    )

    out = bytearray()
    out.extend(b"%PDF-1.4\n")
    out.extend(b"%\xe2\xe3\xcf\xd3\n")

    offsets = [0]
    for obj_id, content in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{obj_id} 0 obj\n".encode("latin-1"))
        out.extend(content)
        out.extend(b"\nendobj\n")

    xref_start = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))

    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n".encode(
            "latin-1"
        )
    )
    out.extend(f"startxref\n{xref_start}\n%%EOF\n".encode("latin-1"))
    pdf_path.write_bytes(out)


def build_pdf(pdf_path: Path, entries: list[dict[str, str]], generated_at: dt.datetime) -> None:
    lines = build_pdf_lines(entries, generated_at)
    max_lines = max(1, int((PAGE_HEIGHT - (2 * MARGIN)) // LINE_HEIGHT))
    pages = paginate_lines(lines, max_lines)
    write_simple_pdf(pdf_path, pages)


def write_run_report(
    run_dir: Path,
    entries: list[dict[str, str]],
    errors: list[str],
    generated_at: dt.datetime,
) -> None:
    report = {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "entry_count": len(entries),
        "error_count": len(errors),
        "errors": errors,
    }
    (run_dir / "run.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()

    opml_path = Path(args.opml).resolve()
    output_root = Path(args.output_dir).resolve()
    state_path = Path(args.state_file).resolve()

    if not opml_path.exists():
        print(f"[error] OPML file not found: {opml_path}")
        return 1

    feeds = parse_opml(opml_path)
    if not feeds:
        print(f"[error] No feeds found in OPML: {opml_path}")
        return 1

    seen_keys = load_state(state_path)
    workers = max(1, args.workers)
    print(
        f"[info] Start fetching {len(feeds)} feeds (workers={workers}, timeout={max(args.timeout, 1)}s)",
        flush=True,
    )
    generated_at = dt.datetime.now()
    run_dir = output_root / generated_at.strftime("%Y-%m-%d")
    run_dir.mkdir(parents=True, exist_ok=True)

    entries, errors, new_seen = collect_new_entries(
        feeds=feeds,
        seen_keys=seen_keys,
        max_per_feed=max(args.max_per_feed, 1),
        timeout_sec=max(args.timeout, 1),
        summary_chars=max(args.summary_chars, 120),
        workers=workers,
    )

    pdf_path = run_dir / "rss_digest.pdf"
    build_pdf(pdf_path, entries, generated_at)
    write_run_report(run_dir, entries, errors, generated_at)

    merged_seen = seen_keys.union(new_seen)
    save_state(state_path, merged_seen)

    print(f"[ok] Generated: {pdf_path}")
    print(f"[ok] New entries: {len(entries)}")
    print(f"[ok] Feeds: {len(feeds)}")
    print(f"[ok] Errors: {len(errors)}")
    if errors:
        print("[warn] See run.json for fetch errors.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

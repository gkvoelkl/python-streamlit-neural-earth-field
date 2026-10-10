"""MkDocs hook: links between the two language versions.

In the Markdown files, a German page links to its English counterpart as
``../en/<name>.md`` and vice versa ``../de/<name>.md``. That works on GitHub.
On the website, each language is built separately (German at the root,
English under ``en/``), so these links would point nowhere. This hook rewrites
them to absolute URLs of the built pages (based on ``site_url``), so MkDocs
does not try to resolve them inside the current language.
"""

from __future__ import annotations

import re

CROSS = re.compile(r"\]\(\.\./(de|en)/([^)#\s]+?)\.md(#[^)\s]*)?\)")


def _root(site_url: str) -> str:
    root = site_url if site_url.endswith("/") else site_url + "/"
    return root[: -len("en/")] if root.endswith("/en/") else root


def on_page_markdown(markdown, page, config, files):
    root = _root(config["site_url"] or "/")

    def target(m: re.Match) -> str:
        lang, name, anchor = m.group(1), m.group(2), m.group(3) or ""
        page_name = "index" if name == "README" else name
        prefix = "en/" if lang == "en" else ""
        return f"]({root}{prefix}{page_name}.html{anchor})"

    return CROSS.sub(target, markdown)

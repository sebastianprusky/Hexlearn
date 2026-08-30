from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
    "connect-src http://127.0.0.1:8000 http://localhost:8000; "
    "media-src 'self' blob:; object-src 'none'; base-uri 'none'"
)


def remove_external_scripts(content: str) -> str:
    content = re.sub(
        r"\s*<link[^>]+fonts\.googleapis\.com[^>]*>",
        "",
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(
        r"\s*<script[^>]+(?:adsbygoogle|googletagmanager|google-analytics)[^>]*>.*?</script>",
        "",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    content = re.sub(
        r"\s*<script[^>]+(?:adsbygoogle|googletagmanager|google-analytics)[^>]*/?>",
        "",
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(
        r"\s*<script>\s*\(function\(i,s,o,g,r,a,m\).*?</script>",
        "",
        content,
        flags=re.DOTALL,
    )
    content = re.sub(
        r"href\s*=\s*(['\"])https?://.*?\1",
        'href="#"',
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    content = re.sub(
        r"\s+onclick=(['\"])window\.open\(.*?\)\1",
        "",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    csp_meta = (
        '<meta http-equiv="Content-Security-Policy" '
        f'content="{CSP}">'
    )
    if "Content-Security-Policy" in content:
        content = re.sub(
            r'<meta\s+http-equiv="Content-Security-Policy"\s+content="[^"]*">',
            csp_meta,
            content,
            count=1,
            flags=re.IGNORECASE,
        )
    else:
        content = content.replace(
            "<head>",
            f"<head>\n\t\t{csp_meta}",
            1,
        )
    return content


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 prepare_clean_copy.py PATH_TO_HEXTRIS")
    target = Path(sys.argv[1]).resolve()
    index = target / "index.html"
    main_script = target / "js" / "main.js"
    if not index.exists() or not main_script.exists():
        raise SystemExit(f"Hextris checkout not found under {target}")

    index_content = remove_external_scripts(index.read_text(encoding="utf-8"))
    index_content = index_content.replace("http://hextris.github.io", "#")
    index.write_text(index_content, encoding="utf-8")

    script_content = main_script.read_text(encoding="utf-8")
    script_content = re.sub(
        r"\s*\$\.get\(['\"]http://54\.183\.184\.126/['\"]\s*\+\s*String\(score\)\)\s*;?",
        "",
        script_content,
    )
    script_content = re.sub(
        r"\s*\(function\(\)\{\s*var script = document\.createElement\('script'\);\s*"
        r"script\.src = 'http://hextris\.io/a\.js';\s*document\.head\.appendChild\(script\);\s*\}\)\(\)",
        "",
        script_content,
        flags=re.DOTALL,
    )
    main_script.write_text(script_content, encoding="utf-8")

    initialization_script = target / "js" / "initialization.js"
    initialization_content = initialization_script.read_text(encoding="utf-8")
    initialization_content = re.sub(
        r"\s*\(function\(i, s, o, g, r, a, m\).*?ga\('send', 'pageview'\);",
        "",
        initialization_content,
        flags=re.DOTALL,
    )
    initialization_script.write_text(initialization_content, encoding="utf-8")

    commit = "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    metadata = {
        "upstream": "https://github.com/Hextris/hextris",
        "commit": commit,
        "telemetryAdded": False,
        "removed": ["analytics", "advertising", "remote score submission", "external runtime requests"],
    }
    (target / ".playlens-clean.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared clean Hextris copy at {target} ({commit[:12]})")


if __name__ == "__main__":
    main()

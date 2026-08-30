from __future__ import annotations


PLAYLENS_LOADER = """
		<!-- PlayLens local fallback waits for the Chrome extension before claiming capture. -->
		<script src="/playlens/core.js?v=064"></script>
		<script src="/playlens/local-bootstrap.js?v=064"></script>
"""


def inject_playlens_loader(html: str) -> str:
    if 'src="/playlens/local-bootstrap.js?v=064"' in html:
        return html
    if "</body>" not in html:
        raise ValueError("Hextris index is missing </body>")
    return html.replace("</body>", f"{PLAYLENS_LOADER}\t</body>", 1)

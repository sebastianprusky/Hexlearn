from __future__ import annotations

import shutil
import sys
from pathlib import Path


SCRIPT_TAG = "\t\t<script type='text/javascript' src='playlens-telemetry.js'></script>"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 install_bridge.py PATH_TO_HEXTRIS")
    target = Path(sys.argv[1]).resolve()
    index = target / "index.html"
    if not index.exists():
        raise SystemExit(f"Hextris index not found: {index}")
    bridge = Path(__file__).with_name("playlens-telemetry.js")
    shutil.copy2(bridge, target / bridge.name)
    content = index.read_text(encoding="utf-8")
    if SCRIPT_TAG not in content:
        if "</body>" not in content:
            raise SystemExit(f"Could not find </body> in {index}")
        content = content.replace("</body>", f"{SCRIPT_TAG}\n</body>", 1)
        index.write_text(content, encoding="utf-8")
    print(f"Installed PlayLens telemetry bridge in {target}")


if __name__ == "__main__":
    main()

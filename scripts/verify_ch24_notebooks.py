"""Execute the eleven Chapter 24 notebooks with API credentials and networking disabled."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OFFLINE_SETUP = """
import os
import socket
import dotenv
NETWORK_ATTEMPTS = []
for name in tuple(os.environ):
    if name.endswith(("_API_KEY", "_API_TOKEN")) or name in ("LLM_PROVIDER", "LLM_MODEL"):
        os.environ.pop(name, None)
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"
dotenv.load_dotenv = lambda *args, **kwargs: False
def block_network(self, address):
    NETWORK_ATTEMPTS.append(str(address))
    raise RuntimeError("Networking disabled by notebook verification")
socket.socket.connect = block_network
get_ipython().execution_count = 0
"""


def verify(path: Path, write: bool = False) -> None:
    notebook = nbformat.read(path, as_version=4)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(OFFLINE_SETUP))
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            'assert not NETWORK_ATTEMPTS, f"Notebook attempted networking: {NETWORK_ATTEMPTS}"'
        )
    )
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.create_kernel_manager()
    client.km.kernel_spec.argv[0] = sys.executable
    client.execute()
    notebook.cells = notebook.cells[1:-1]
    if write:
        nbformat.write(notebook, path)
    print(f"PASS {path.relative_to(ROOT)} (offline)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--write", action="store_true", help="Save freshly executed outputs")
    args = parser.parse_args()
    paths = args.paths or sorted((ROOT / "24_autonomous_agents").glob("[0-9]*.ipynb"))
    for path in paths:
        verify(path.resolve(), args.write)


if __name__ == "__main__":
    main()

"""Custom exceptions for ML4T data infrastructure.

Provides clear, actionable error messages when data is missing or unavailable.
All exceptions point readers to the appropriate documentation.
"""

import os
from pathlib import Path

_DATA_ROOT_VARIABLE = "ML4T_DATA_PATH"


def _dotenv_data_path(repo_root: Path) -> str | None:
    """Return the data path assigned in .env, or None when it assigns none."""
    env_file = repo_root / ".env"
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{_DATA_ROOT_VARIABLE}="):
            return stripped.split("=", 1)[1].strip().strip("\"'")
    return None


def _resolved_data_root() -> tuple[Path, str] | None:
    """Return the data root every loader reads under, and where it came from.

    `utils.config` resolves the root once, from the environment, then .env, then the
    repository's own `data/`. Reporting which of the three won is what separates "you
    have not downloaded this" from "you are pointed at a directory that does not hold
    it", and the two need different actions from the reader.
    """
    try:
        from utils.config import ML4T_DATA_PATH, REPO_ROOT
    except Exception:
        return None

    root = Path(ML4T_DATA_PATH)
    # utils.config loads .env into the environment, so a set variable proves nothing
    # about which source set it. Compare against .env to tell them apart.
    from_dotenv = _dotenv_data_path(Path(REPO_ROOT))
    if os.environ.get(_DATA_ROOT_VARIABLE) is None:
        return root, (
            f"{_DATA_ROOT_VARIABLE} is not set, so it defaults to the repository's own "
            f"data/ directory."
        )
    if from_dotenv is not None and Path(from_dotenv).expanduser() == root:
        return root, f"Set by {_DATA_ROOT_VARIABLE} in .env."
    return root, f"Set by the {_DATA_ROOT_VARIABLE} environment variable."


class DataNotFoundError(FileNotFoundError):
    """Raised when a required dataset is not found.

    Provides clear instructions for obtaining the missing data.

    Args:
        dataset_name: Human-readable name of the dataset (e.g., "ETF Universe")
        path: Path where the data was expected
        download_script: Name of the download script (e.g., "yfinance_etfs.py")
        requires_api_key: Name of required API key if applicable
        instructions: Custom multi-line instructions (overrides download_script template)
        download_url: Hosted-download URL (e.g., AlgoSeek S3 bucket)
        derivation_notebook: Path/stem of a notebook that derives this dataset,
            for derived datasets where readers may want to follow the computation
        readme: Repo-relative path to the dataset's local README with full
            download instructions (e.g., "data/equities/fundamentals/README.md").
            Falls back to "data/README.md" when not set.

    Example:
        >>> raise DataNotFoundError(
        ...     dataset_name="ETF Universe",
        ...     path=Path("/data/etfs/market/etf_universe.parquet"),
        ...     download_script="yfinance_etfs.py"
        ... )
    """

    def __init__(
        self,
        dataset_name: str,
        path: Path | str,
        download_script: str | None = None,
        requires_api_key: str | None = None,
        instructions: str | None = None,
        download_url: str | None = None,
        derivation_notebook: str | None = None,
        readme: str | None = None,
    ):
        self.dataset_name = dataset_name
        self.path = Path(path)
        self.download_script = download_script
        self.requires_api_key = requires_api_key
        self.instructions = instructions
        self.download_url = download_url
        self.derivation_notebook = derivation_notebook
        self.readme = readme

        message = self._build_message()
        super().__init__(message)

    def _data_root_lines(self) -> list[str]:
        """Name the data root the expected location was resolved under.

        Without this, a reader whose data sits on another drive is told to download
        it again, and nothing in the message names the variable that would point at
        the copy they already have.
        """
        resolved = _resolved_data_root()
        if resolved is None:
            return []
        root, source = resolved
        try:
            if not self.path.is_relative_to(root):
                return []
        except (OSError, ValueError):
            return []
        return [
            f"Data root: {root}",
            f"  {source}",
            f"  If you already have this data elsewhere, point {_DATA_ROOT_VARIABLE} at",
            "  that directory - export it, or set it in .env - rather than downloading",
            "  it again.",
            "",
        ]

    def _build_message(self) -> str:
        lines = [
            "",
            "=" * 70,
            f"DATA NOT FOUND: {self.dataset_name}",
            "=" * 70,
            "",
            f"Expected location: {self.path}",
            "",
            *self._data_root_lines(),
        ]

        if self.instructions:
            lines.extend([self.instructions, ""])
        elif self.download_url:
            lines.extend(
                [
                    "Download from:",
                    f"  {self.download_url}",
                    "",
                    f"Extract to: {self.path.parent}",
                    "",
                ]
            )
        elif self.download_script:
            lines.extend(
                [
                    "To download this dataset (from repo root):",
                    f"  uv run python {self.download_script}",
                    "",
                ]
            )

        if self.derivation_notebook:
            lines.extend(
                [
                    f"How this dataset is built: {self.derivation_notebook}",
                    "",
                ]
            )

        if self.requires_api_key:
            lines.extend(
                [
                    f"Note: Requires {self.requires_api_key} in .env file",
                    "",
                ]
            )

        readme_path = self.readme or "data/README.md"
        lines.extend(
            [
                f"For complete instructions, see: {readme_path}",
                "=" * 70,
            ]
        )

        return "\n".join(lines)


class DownloadError(RuntimeError):
    """Raised when a data download fails.

    Used by download scripts to provide clear failure messages instead of
    silently succeeding with empty or incomplete data.

    Args:
        dataset_name: Human-readable name of the dataset
        reason: Explanation of why the download failed
        suggestion: Optional suggestion for resolving the issue
    """

    def __init__(
        self,
        dataset_name: str,
        reason: str,
        suggestion: str | None = None,
    ):
        self.dataset_name = dataset_name
        self.reason = reason
        self.suggestion = suggestion

        message = self._build_message()
        super().__init__(message)

    def _build_message(self) -> str:
        lines = [
            "",
            "=" * 70,
            f"DOWNLOAD FAILED: {self.dataset_name}",
            "=" * 70,
            "",
            f"Reason: {self.reason}",
            "",
        ]

        if self.suggestion:
            lines.extend(
                [
                    f"Suggestion: {self.suggestion}",
                    "",
                ]
            )

        lines.extend(
            [
                "For help, see: data/README.md",
                "=" * 70,
            ]
        )

        return "\n".join(lines)


class MissingDependencyError(ImportError):
    """Raised when a required dependency is not installed.

    Provides clear installation instructions.

    Args:
        package: Name of the missing package
        install_command: Command to install the package
        purpose: What the package is needed for
    """

    def __init__(
        self,
        package: str,
        install_command: str | None = None,
        purpose: str | None = None,
    ):
        self.package = package
        self.install_command = install_command or f"pip install {package}"
        self.purpose = purpose

        message = self._build_message()
        super().__init__(message)

    def _build_message(self) -> str:
        lines = [
            "",
            f"Missing dependency: {self.package}",
        ]

        if self.purpose:
            lines.append(f"Required for: {self.purpose}")

        lines.extend(
            [
                "",
                f"Install with: {self.install_command}",
            ]
        )

        return "\n".join(lines)

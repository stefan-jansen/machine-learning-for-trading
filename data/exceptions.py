"""Custom exceptions for ML4T data infrastructure.

Provides clear, actionable error messages when data is missing or unavailable.
All exceptions point readers to the appropriate documentation.
"""

from pathlib import Path


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

    def _build_message(self) -> str:
        lines = [
            "",
            "=" * 70,
            f"DATA NOT FOUND: {self.dataset_name}",
            "=" * 70,
            "",
            f"Expected location: {self.path}",
            "",
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

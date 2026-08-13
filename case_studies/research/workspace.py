from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.create_experiment import create_experiment
from utils.paths import REPO_ROOT

from .contracts import ExecutionTier

if TYPE_CHECKING:
    from .labels import LabelCatalog
    from .lifecycle import Lifecycle
    from .results import ResultsCatalog
    from .strategy import Strategy


_ACTIVE_OUTPUT_ROOT: Path | None = None


def _release_manifest_digest(case_dir: Path) -> str:
    release_manifest = case_dir / "run_log" / ".release" / "SHA256SUMS"
    if release_manifest.exists():
        return hashlib.sha256(release_manifest.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for path in sorted(p for p in case_dir.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(case_dir)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _source_commit(release_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(release_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _clear_root_sensitive_caches() -> None:
    from case_studies.utils import backtest_loaders, cv_window
    from case_studies.utils.registry import specs
    from utils import artifact_specs

    for cached in (
        artifact_specs._load_setup_config_cached,
        artifact_specs._load_market_data_spec_cached,
        artifact_specs._load_label_spec_cached,
        artifact_specs._load_feature_spec_cached,
        cv_window._load_setup_yaml,
        cv_window._fold_splits,
        cv_window._holdout_window,
        backtest_loaders._load_backtest_preset_config,
        backtest_loaders._load_case_setup_yaml,
    ):
        cached.cache_clear()
    specs._CONFIG_DIR = None


@dataclass(frozen=True)
class Study:
    case_study: str
    root: Path
    release_root: Path
    output_root: Path | None
    read_only: bool
    manifest: dict

    @classmethod
    def open(
        cls,
        case_study: str,
        workspace: str | Path | None = None,
        *,
        release_root: str | Path = REPO_ROOT,
    ) -> Study:
        release_root = Path(release_root).expanduser().resolve()
        release_case_dir = release_root / "case_studies" / case_study
        if not release_case_dir.is_dir():
            raise FileNotFoundError(f"Unknown released case study: {release_case_dir}")

        if workspace is None:
            study = cls(
                case_study=case_study,
                root=release_case_dir,
                release_root=release_root,
                output_root=None,
                read_only=True,
                manifest={
                    "schema_version": 1,
                    "case_study": case_study,
                    "baseline_source_commit": _source_commit(release_root),
                    "baseline_manifest_sha256": _release_manifest_digest(release_case_dir),
                },
            )
            study.activate()
            return study

        output_root = Path(workspace).expanduser().resolve()
        target = output_root / case_study
        manifest_path = target / ".study.json"
        if target.exists():
            if not manifest_path.is_file():
                raise ValueError(f"Existing workspace has no .study.json manifest: {target}")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("schema_version") != 1 or manifest.get("case_study") != case_study:
                raise ValueError(f"Invalid workspace manifest: {manifest_path}")
        else:
            manifest = {
                "schema_version": 1,
                "case_study": case_study,
                "baseline_source_commit": _source_commit(release_root),
                "baseline_manifest_sha256": _release_manifest_digest(release_case_dir),
                "created_at": datetime.now(UTC).isoformat(),
            }
            create_experiment(
                case_study,
                output_root,
                repo_root=release_root,
                manifest=manifest,
            )
        study = cls(
            case_study=case_study,
            root=target,
            release_root=release_root,
            output_root=output_root,
            read_only=False,
            manifest=manifest,
        )
        from case_studies.utils.registry.store import _open_registry

        _open_registry(target).close()
        study.activate()
        return study

    def activate(self, execution_tier: str | ExecutionTier = ExecutionTier.CANONICAL) -> Path:
        global _ACTIVE_OUTPUT_ROOT
        tier = ExecutionTier(execution_tier)
        if self.read_only:
            os.environ.pop("ML4T_OUTPUT_DIR", None)
            _ACTIVE_OUTPUT_ROOT = None
            _clear_root_sensitive_caches()
            return self.root

        base_output_root = self.output_root
        assert base_output_root is not None
        output_root = base_output_root
        if tier is ExecutionTier.PREVIEW:
            output_root = output_root / ".preview"
            preview_case = output_root / self.case_study
            if not preview_case.exists():
                preview_case.mkdir(parents=True)
                shutil.copytree(self.root / "config", preview_case / "config")
                shared_config = base_output_root / "config"
                if shared_config.exists():
                    shutil.copytree(shared_config, output_root / "config", dirs_exist_ok=True)
        if output_root != _ACTIVE_OUTPUT_ROOT:
            os.environ["ML4T_OUTPUT_DIR"] = str(output_root)
            _ACTIVE_OUTPUT_ROOT = output_root
            _clear_root_sensitive_caches()
        return output_root / self.case_study

    def storage_root(self, execution_tier: str | ExecutionTier = ExecutionTier.CANONICAL) -> Path:
        tier = ExecutionTier(execution_tier)
        if tier is ExecutionTier.PREVIEW:
            if self.read_only:
                raise PermissionError("read-only release cannot create preview results")
            assert self.output_root is not None
            return self.output_root / ".preview" / self.case_study
        return self.root

    def require_writable(self) -> None:
        if self.read_only:
            raise PermissionError("released baseline is read-only; open a workspace to write")

    @property
    def labels(self) -> LabelCatalog:
        from .labels import LabelCatalog

        return LabelCatalog(self)

    @property
    def results(self) -> ResultsCatalog:
        from .results import ResultsCatalog

        return ResultsCatalog(self)

    @property
    def lifecycle(self) -> Lifecycle:
        from .lifecycle import Lifecycle

        return Lifecycle(self)

    def strategy(self, **request) -> Strategy:
        from .strategy import Strategy

        return Strategy.from_request(self, request)

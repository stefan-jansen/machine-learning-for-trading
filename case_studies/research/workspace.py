from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.create_experiment import create_experiment
from utils.paths import REPO_ROOT

from .contracts import ExecutionTier

if TYPE_CHECKING:
    from .catalog import BacktestCatalog, PredictionCatalog
    from .causal import CausalRequest
    from .labels import LabelCatalog
    from .models import ModelRequest
    from .recovery import ExecutionLedger
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


def _resolved_directory_symlink(path: Path) -> Path | None:
    if not path.is_symlink():
        return None
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    return resolved if resolved.is_dir() else None


def _ensure_input_link(preview_case: Path, source: Path) -> None:
    """Point `<preview>/<name>` at the active study's input directory.

    The link belongs to whichever study is currently active, so a link left by a previous
    study is repointed rather than refused. This used to raise, which made a second study
    previewing into one workspace fail on activation - and refusing was never the safer
    half: leaving the stale link would have had the preview read the previous study's
    labels under the current study's name. `_ensure_config_link` below reached the same
    conclusion for `config`, and the two disagreeing was the defect.
    """
    resolved = source.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"preview input is not a directory: {source}")
    link = preview_case / source.name
    if _resolved_directory_symlink(link) == resolved:
        return
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise ValueError(f"preview input path must be a directory symlink: {link}")
    link.symlink_to(resolved, target_is_directory=True)


def _ensure_config_link(link: Path, source: Path) -> None:
    resolved = source.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"preview config is not a directory: {source}")
    if _resolved_directory_symlink(link) == resolved:
        return
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        if not link.is_dir():
            raise ValueError(f"preview config path is not a directory: {link}")
        backup = link.with_name(f".{link.name}.{uuid.uuid4().hex}.stale")
        link.rename(backup)
        try:
            link.symlink_to(resolved, target_is_directory=True)
        except Exception:
            backup.rename(link)
            raise
        shutil.rmtree(backup)
        return
    link.symlink_to(resolved, target_is_directory=True)


@dataclass(frozen=True)
class Study:
    case_study: str
    root: Path
    release_root: Path
    output_root: Path | None
    read_only: bool
    manifest: dict
    # The notebook that opened this study, recorded on every training run it registers so the
    # registry can answer "which notebook wrote this". Stated by the caller rather than inferred:
    # under papermill the executing file is a temp .ipynb and `__file__` may be absent entirely,
    # so a frame walk is wrong exactly where it would be needed.
    entry_point: str | None = None
    # The tier this study was opened for, held rather than re-derived. `ML4T_OUTPUT_DIR` is
    # process-global and `activate` never clears it, so every consumer that read the tier from
    # the environment was answering "is a preview active anywhere in this process" while asking
    # "is this study a preview". Those differ whenever one process holds two studies, and they
    # differ in both directions - see the two failures this closes.
    execution_tier: ExecutionTier = ExecutionTier.CANONICAL

    # Set only by `Study.at`, where the released case directory is the one root the study was
    # given and is not derivable from `release_root`: a fixture root, or any output directory
    # that is not `<repo>/case_studies/<name>`.
    release_case_dir: Path | None = None

    @classmethod
    def at(
        cls,
        case_dir: str | Path,
        *,
        case_study: str | None = None,
        entry_point: str | None = None,
    ) -> Study:
        """A read-only study over one registry root, which never activates.

        Every other way in ends in `activate()`, which rewrites `ML4T_OUTPUT_DIR` process-wide
        and clears the root-sensitive caches. For a notebook that *produces* results that is the
        point - it decides where writes land. For one that only reads, it silently moves every
        subsequent path lookup to a different case directory than the one the notebook resolved,
        and `open_study(execution_tier="preview")` moves it to `.preview/<case>`, whose registry
        is created empty. An analysis notebook pointed there does not fail; it reports on nothing.

        This form takes the directory the caller already resolved and answers for that registry
        alone: `root` and `release_case_root` are both `case_dir`, so `OfficialPopulation.one`
        and `Result.open` read it and nothing else.
        """
        case_dir = Path(case_dir).expanduser().resolve()
        return cls(
            case_study=case_study or case_dir.name,
            root=case_dir,
            release_root=REPO_ROOT,
            output_root=None,
            read_only=True,
            manifest={
                "schema_version": 1,
                "case_study": case_study or case_dir.name,
                "single_root": str(case_dir),
            },
            entry_point=entry_point,
            release_case_dir=case_dir,
        )

    @classmethod
    def open(
        cls,
        case_study: str,
        workspace: str | Path | None = None,
        *,
        release_root: str | Path = REPO_ROOT,
        entry_point: str | None = None,
        execution_tier: str | ExecutionTier = ExecutionTier.CANONICAL,
    ) -> Study:
        execution_tier = ExecutionTier(execution_tier)
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
                entry_point=entry_point,
                execution_tier=execution_tier,
            )
            study.activate()
            return study

        output_root = Path(workspace).expanduser().resolve()
        target = output_root / case_study
        manifest_path = target / ".study.json"
        if target.exists():
            if not manifest_path.is_file():
                isolated_root = os.environ.get("ML4T_OUTPUT_DIR")
                may_adopt = (
                    isolated_root is not None
                    and Path(isolated_root).expanduser().resolve() == output_root
                    and not target.is_symlink()
                    and (target / "config").is_dir()
                )
                if not may_adopt:
                    raise ValueError(f"Existing workspace has no .study.json manifest: {target}")
                manifest = {
                    "schema_version": 1,
                    "case_study": case_study,
                    "baseline_source_commit": _source_commit(release_root),
                    "baseline_manifest_sha256": _release_manifest_digest(release_case_dir),
                    "created_at": datetime.now(UTC).isoformat(),
                    "adopted_output_root": True,
                }
                manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            else:
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
                include_release_run_log=False,
            )
        study = cls(
            case_study=case_study,
            root=target,
            release_root=release_root,
            output_root=output_root,
            read_only=False,
            manifest=manifest,
            entry_point=entry_point,
            execution_tier=execution_tier,
        )
        from case_studies.utils.registry.store import _open_registry

        _open_registry(target).close()
        study.activate()
        return study

    @classmethod
    def regenerate(
        cls,
        case_study: str,
        *,
        release_root: str | Path = REPO_ROOT,
        entry_point: str | None = None,
    ) -> Study:
        """Open the canonical generated-artifact links for maintainer regeneration."""
        release_root = Path(release_root).expanduser().resolve()
        case_dir = release_root / "case_studies" / case_study
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Unknown released case study: {case_dir}")
        generated = (case_dir / "features", case_dir / "labels", case_dir / "run_log")
        invalid = [path for path in generated if _resolved_directory_symlink(path) is None]
        if invalid:
            raise PermissionError(
                f"canonical regeneration requires generated-artifact directory symlinks: {invalid}"
            )
        study = cls(
            case_study=case_study,
            root=case_dir,
            release_root=release_root,
            output_root=case_dir.parent,
            read_only=False,
            entry_point=entry_point,
            manifest={
                "schema_version": 1,
                "case_study": case_study,
                "baseline_source_commit": _source_commit(release_root),
                "baseline_manifest_sha256": _release_manifest_digest(case_dir),
                "regeneration": True,
            },
        )
        study.activate()
        return study

    def activate(self, execution_tier: str | ExecutionTier | None = None) -> Path:
        global _ACTIVE_OUTPUT_ROOT
        tier = self.execution_tier if execution_tier is None else ExecutionTier(execution_tier)
        if self.read_only:
            # Point the path helpers at the root this study answers for, rather than
            # clearing the variable. `get_case_study_dir` reads `ML4T_OUTPUT_DIR` and
            # falls back to the repo's own `case_studies/` when it is absent, so popping
            # it moved every later lookup - `LabelCatalog.get`'s label artifact included -
            # off the directory the caller resolved and onto the repo, for the rest of the
            # process. Under a redirected output root that directory holds nothing, and
            # the failure surfaces as a missing artifact rather than as a moved root:
            # sp500_equity_option_analytics' 20_strategy_analysis failed that way, through
            # the latent-factor holdout hook.
            #
            # `get_case_study_dir` composes `ML4T_OUTPUT_DIR / case_study`, so the value is
            # this root's parent, and only where the root is actually named for its case
            # study. Where it is not, the variable is left as it stands: whatever the caller
            # set is closer to right than the repo fallback.
            if self.root.name == self.case_study:
                output_root = self.root.parent
                if output_root != _ACTIVE_OUTPUT_ROOT:
                    os.environ["ML4T_OUTPUT_DIR"] = str(output_root)
                    _ACTIVE_OUTPUT_ROOT = output_root
            _clear_root_sensitive_caches()
            return self.root

        base_output_root = self.output_root
        assert base_output_root is not None
        output_root = base_output_root
        if tier is ExecutionTier.PREVIEW:
            output_root = output_root / ".preview"
            preview_case = output_root / self.case_study
            preview_case.mkdir(parents=True, exist_ok=True)
            _ensure_config_link(preview_case / "config", self.root / "config")
            shared_config = base_output_root / "config"
            if shared_config.exists():
                _ensure_config_link(output_root / "config", shared_config)
            for name in ("labels", "features"):
                source = self.root / name
                if source.exists():
                    _ensure_input_link(preview_case, source)
            from case_studies.utils.registry.store import _open_registry

            _open_registry(preview_case).close()
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
    def predictions(self) -> PredictionCatalog:
        from .catalog import PredictionCatalog

        return PredictionCatalog(self)

    @property
    def backtests(self) -> BacktestCatalog:
        from .catalog import BacktestCatalog

        return BacktestCatalog(self)

    @property
    def release_root_is_immutable(self) -> bool:
        """Whether the released case directory is guaranteed not to change while this is open.

        A released root is a checkout, so a reader may tell SQLite the file is immutable and
        skip locking. A `Study.at` root is whatever directory the caller named and can still be
        written - so that promise would be false, and a read under it can miss rows committed
        after the study was constructed.
        """
        return self.release_case_dir is None

    @property
    def release_case_root(self) -> Path:
        if self.release_case_dir is not None:
            return self.release_case_dir
        return self.release_root / "case_studies" / self.case_study

    @property
    def executions(self) -> ExecutionLedger:
        from .recovery import ExecutionLedger

        return ExecutionLedger(self)

    def strategy(self, **request) -> Strategy:
        from .strategy import Strategy

        return Strategy.from_request(self, request)

    def model(self, **request) -> ModelRequest:
        from .models import ModelRequest

        return ModelRequest.from_request(self, request)

    def causal(self, **request) -> CausalRequest:
        from .causal import CausalRequest

        return CausalRequest.from_request(self, request)


def open_study(
    case_study: str,
    *,
    execution_tier: str | ExecutionTier = ExecutionTier.CANONICAL,
    workspace: str | Path | None = None,
    release_root: str | Path = REPO_ROOT,
    entry_point: str | None = None,
) -> Study:
    """Open the study a notebook should execute against for its tier.

    Canonical execution with no workspace regenerates the case study's own artifacts in place, and
    is the production path. Canonical execution *with* a workspace is the same computation at full
    scale writing to an isolated registry - a rehearsal that can be compared against the published
    result without being able to damage it. Preview reads the same inputs, writes only to
    ``workspace``, and must declare the reductions that make it cheap.
    """
    tier = ExecutionTier(execution_tier)
    release_root = Path(release_root).expanduser().resolve()
    if tier is ExecutionTier.CANONICAL:
        if workspace is None:
            return Study.regenerate(case_study, release_root=release_root, entry_point=entry_point)
        return Study.open(
            case_study,
            workspace=workspace,
            release_root=release_root,
            entry_point=entry_point,
        )

    if workspace is None:
        raise ValueError("preview execution requires an explicit workspace")
    workspace = Path(workspace).expanduser().resolve()
    case_dir = release_root / "case_studies" / case_study
    generated = tuple(case_dir / name for name in ("features", "labels", "run_log"))
    if not all(path.is_symlink() for path in generated):
        return Study.open(
            case_study,
            workspace=workspace,
            release_root=release_root,
            entry_point=entry_point,
            execution_tier=tier,
        )

    # A maintainer worktree links its generated directories to shared data, which
    # `create_experiment` cannot copy. Read those inputs in place and redirect every write.
    workspace.mkdir(parents=True, exist_ok=True)
    shared_config = workspace / "config"
    if not shared_config.exists():
        shared_config.symlink_to(release_root / "case_studies" / "config", target_is_directory=True)
    study = Study(
        case_study=case_study,
        root=case_dir,
        release_root=release_root,
        output_root=workspace,
        read_only=False,
        entry_point=entry_point,
        manifest={
            "schema_version": 1,
            "case_study": case_study,
            "baseline_source_commit": _source_commit(release_root),
            "preview_only": True,
        },
        execution_tier=tier,
    )
    study.activate()
    return study

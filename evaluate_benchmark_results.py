#!/usr/bin/env python3
"""Post‑hoc benchmark evaluator with execution log analysis.

Accepts a directory containing benchmark archives (or a single archive),
extracts each session, runs static checks (file existence, grep, pytest),
and analyzes execution_logs.json to extract thread behaviour metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatch
from importlib import import_module
from pathlib import Path
from pathlib import PurePosixPath
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

from utils.env_loader import load_dotenv_file

# ----------------------------------------------------------------------
# 1. Benchmark static checks (original code)
# ----------------------------------------------------------------------

BENCHMARK_NAMES = [
    "benchmark_task_api",
    "benchmark_csv_pipeline",
    "benchmark_chat_server",
    "benchmark_url_shortener",
    "benchmark_expense_tracker",
]

OPIK_SCORE_NAME = "benchmark_requirements_rating"
OPIK_NO_STUBS_SCORE_NAME = "benchmark_completeness"
DEFAULT_OPIK_HOST = "https://www.comet.com/opik/api"
NO_STUBS_CHECK_ID = "NO_STUBS"

opik = None


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    description: str
    kind: str
    passed: bool
    details: str
    weight: float = 1.0
    duration_seconds: float = 0.0
    match_count: int | None = None


@dataclass(frozen=True)
class CheckSpec:
    check_id: str
    description: str
    kind: str
    weight: float = 1.0
    pattern: str | None = None
    must_absent: bool = False
    paths: tuple[str, ...] = ()
    command: tuple[str, ...] | None = None
    command_cwd: str | None = None
    timeout_seconds: int = 120


def _benchmark_checks() -> dict[str, list[CheckSpec]]:
    # Placeholder detection pattern (must_absent=True)
    placeholder_pattern = r"^[ \t]*pass[ \t]*$|raise\s+NotImplementedError|#\s*(TODO|FIXME|stub|placeholder)"
    no_stubs_check = CheckSpec(
        NO_STUBS_CHECK_ID,
        "No placeholder/stub code",
        "grep",
        pattern=placeholder_pattern,
        must_absent=True,
        paths=("*.py", "**/*.py"),
    )

    return {
        "benchmark_task_api": [
            CheckSpec("AC1", "Project contains a Dockerfile", "file_exists", paths=("Dockerfile",)),
            CheckSpec("AC2", "FastAPI is used", "grep", pattern=r"from\s+fastapi|import\s+fastapi"),
            CheckSpec("AC3", "SQLite is used", "grep", pattern=r"sqlite"),
            CheckSpec("AC4", "JWT auth is implemented", "grep", pattern=r"jwt|bearer|jose"),
            CheckSpec("AC5", "Soft delete fields appear in the code", "grep", pattern=r"is_deleted|deleted_at"),
            CheckSpec("AC6", "Pagination keywords appear in routes", "grep", pattern=r"skip|limit|page", paths=("routes/**", "**/routes/**", "**/*route*.*", "**/*router*.*")),
            CheckSpec("AC7", "User isolation keywords appear in the code", "grep", pattern=r"user_id|owner_id"),
            CheckSpec("AC8", "Pytest suite is runnable", "command", command=(sys.executable, "-m", "pytest", "-q"), timeout_seconds=600),
            no_stubs_check,
        ],
        "benchmark_csv_pipeline": [
            CheckSpec("AC1", "Pipeline source exists", "grep", pattern=r"csv|pandas|argparse"),
            CheckSpec("AC2", "JSON or JSON Lines output logic appears in the code", "grep", pattern=r"json\.dump|json\.dumps|orjson|jsonlines|\.jsonl|JSONL"),
            CheckSpec("AC3", "Streaming/chunked reading appears in the code", "grep", pattern=r"chunksize|chunk_size|iterchunks|readline|for\s+.*\s+in\s+.*csv"),
            CheckSpec("AC4", "Malformed row handling appears in the code", "grep", pattern=r"skip|malformed|invalid|error"),
            CheckSpec("AC5", "Parallel execution is present", "grep", pattern=r"ProcessPoolExecutor|ThreadPoolExecutor|Pool"),
            CheckSpec("AC6", "CLI support is present", "grep", pattern=r"argparse|click|typer"),
            CheckSpec("AC7", "Logging is present", "grep", pattern=r"logging\.|logger\."),
            no_stubs_check,
        ],
        "benchmark_chat_server": [
            CheckSpec("AC1", "Project contains a Dockerfile", "file_exists", paths=("Dockerfile",)),
            CheckSpec("AC2", "WebSocket implementation appears in the code", "grep", pattern=r"WebSocket|websocket|ws://"),
            CheckSpec("AC3", "Persistent storage appears in the code", "grep", pattern=r"INSERT|session\.add|db\.execute"),
            CheckSpec("AC4", "Room support appears in the code", "grep", pattern=r"room|channel|room_id"),
            CheckSpec("AC5", "Message ordering keywords appear in the code", "grep", pattern=r"timestamp|created_at|sequence|ORDER BY"),
            CheckSpec("AC6", "Authentication keywords appear near the WS flow", "grep", pattern=r"token|authenticate|jwt|login"),
            CheckSpec("AC7", "Concurrency support for many users appears in the code", "grep", pattern=r"async\s+def|await|asyncio|connections|100|limit|capacity|semaphore"),
            no_stubs_check,
        ],
        "benchmark_url_shortener": [
            CheckSpec("AC1", "Docker Compose file exists", "file_exists", paths=("docker-compose.yml", "compose.yml")),
            CheckSpec("AC2", "Shortening endpoint keywords appear", "grep", pattern=r"short_code|shorten|custom_code"),
            CheckSpec("AC3", "Redirect logic appears in the code", "grep", pattern=r"redirect|301|302"),
            CheckSpec("AC4", "Custom slug support appears in the code", "grep", pattern=r"custom_code|slug"),
            CheckSpec("AC5", "Expiry handling appears in the code", "grep", pattern=r"expires_at|expired|410"),
            CheckSpec("AC6", "Click counting appears in the code", "grep", pattern=r"click_count|counter|increment"),
            CheckSpec("AC7", "Rate limiting keywords appear in the code", "grep", pattern=r"rate.?limit|429|ttl"),
            CheckSpec("AC8", "PostgreSQL usage appears in the code", "grep", pattern=r"postgresql|psycopg|postgres"),
            CheckSpec("AC9", "Redis usage appears in the code", "grep", pattern=r"redis|Redis"),
            CheckSpec("AC10", "API key authentication appears in the code", "grep", pattern=r"api[_-]?key|x-api-key|APIKey|authorization"),
            no_stubs_check,
        ],
        "benchmark_expense_tracker": [
            CheckSpec("AC1", "CLI support is present", "grep", pattern=r"argparse|click|typer"),
            CheckSpec("AC2", "Expense CRUD keywords appear in the code", "grep", pattern=r"add|edit|delete|expense"),
            CheckSpec("AC3", "Expense categories appear in the code", "grep", pattern=r"category|categories"),
            CheckSpec("AC4", "Fixed-rate currency conversion appears in the code", "grep", pattern=r"currency|exchange|convert|rate"),
            CheckSpec("AC5", "Monthly summary keywords appear in the code", "grep", pattern=r"summary|report|category"),
            CheckSpec("AC6", "Recurring monthly expense handling appears in the code", "grep", pattern=r"recurr|monthly|subscription|auto.?generate"),
            CheckSpec("AC7", "CSV export with optional date filtering appears in the code", "grep", pattern=r"csv|export|date_range|start_date|end_date|from_date|to_date"),
            CheckSpec("AC8", "SQLite usage appears in the code", "grep", pattern=r"sqlite"),
            no_stubs_check,
        ],
    }


def _load_hidden_spec_text(benchmark_name: str) -> str | None:
    yaml_path = Path("yaml_instance/ChatDev_simple_benchmarks_v1.yaml")
    if not yaml_path.exists():
        return None
    try:
        text = yaml_path.read_text(encoding="utf-8")
    except Exception:
        return None
    pattern = re.compile(rf"^{re.escape(benchmark_name)}:\s*\|-\n((?:\s{{4}}.*\n)+)", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return None
    block = m.group(1)
    cleaned = "\n".join(line[4:] if line.startswith("    ") else line for line in block.splitlines())
    return cleaned.strip().lower()


def _should_skip_check_by_hidden_spec(benchmark_name: str, check: CheckSpec, hidden_spec_text: str | None) -> bool:
    if not hidden_spec_text:
        return False
    conditional_requirements: dict[str, dict[str, list[str]]] = {
        "benchmark_chat_server": {"AC3": ["persist", "database", "sqlite", "store", "history"]},
        "benchmark_csv_pipeline": {
            "AC5": ["parallel", "parallel processing", "processpoolexecutor", "threadpoolexecutor"],
            "AC3": ["stream", "streaming", "chunksize", "chunk"],
        },
        "benchmark_task_api": {
            "AC3": ["sqlite", "database", "persist"],
            "AC5": ["soft delete", "is_deleted", "deleted_at"],
            "AC6": ["pagination", "skip", "limit", "page"],
        },
        "benchmark_url_shortener": {"AC8": ["postgres", "postgresql", "psycopg"]},
    }
    bm_map = conditional_requirements.get(benchmark_name)
    if not bm_map:
        return False
    reqs = bm_map.get(check.check_id)
    if not reqs:
        return False
    for kw in reqs:
        if kw.lower() in hidden_spec_text:
            return False
    return True


def _is_completeness_check(result_or_spec: CheckResult | CheckSpec) -> bool:
    return result_or_spec.check_id != NO_STUBS_CHECK_ID


def _score_results(results: Sequence[CheckResult]) -> dict[str, Any]:
    completeness_results = [result for result in results if _is_completeness_check(result)]
    active_completeness_results = [result for result in completeness_results if result.weight > 0]
    no_stubs_result = next((result for result in results if result.check_id == NO_STUBS_CHECK_ID), None)

    completeness_total_weight = sum(result.weight for result in completeness_results)
    completeness_passed_weight = sum(result.weight for result in completeness_results if result.passed)
    completeness_score = (
        100.0 * completeness_passed_weight / completeness_total_weight
        if completeness_total_weight
        else 0.0
    )
    incomplete_stub_count = (
        no_stubs_result.match_count
        if no_stubs_result is not None and no_stubs_result.match_count is not None
        else None
    )

    return {
        "completeness_score": round(completeness_score, 2),
        "completeness_rating": _score_to_rating(completeness_score),
        "passed_checks": sum(1 for result in active_completeness_results if result.passed),
        "total_checks": len(active_completeness_results),
        "no_stubs_passed": no_stubs_result.passed if no_stubs_result is not None else None,
        "no_stubs_details": no_stubs_result.details if no_stubs_result is not None else "not evaluated",
        "incomplete_stub_count": incomplete_stub_count,
    }


# ----------------------------------------------------------------------
# 2. Execution log analysis (thread behaviour metrics)
# ----------------------------------------------------------------------

def to_seconds(timestamp: str) -> float:
    """Convert ISO timestamp to float seconds since epoch."""
    dt = datetime.fromisoformat(timestamp)
    return dt.timestamp()


def parse_execution_logs(log_path: Path) -> List[Dict[str, Any]]:
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("logs", [])


def analyze_logs(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    error_count = 0
    warning_count = 0
    tool_calls = 0
    tool_failures = 0
    cycle_iterations: Dict[str, int] = {}
    task_finished = False

    # Encontrar a última entrada do FINAL que venha do Tester
    last_tester_final = None
    for entry in logs:
        if entry.get("node_id") == "FINAL":
            details = entry.get("details", {})
            if details.get("output_source") == "Tester":
                last_tester_final = entry

    # Se existir e o output contiver "<INFO> Finished", a tarefa está concluída
    if last_tester_final:
        output_text = last_tester_final.get("details", {}).get("output", "")
        if "<INFO> Finished" in output_text:
            task_finished = True

    # Restante análise (erros, warnings, tool calls, ciclos)
    for entry in logs:
        level = entry.get("level")
        if level == "ERROR":
            error_count += 1
        elif level == "WARNING":
            warning_count += 1

        event = entry.get("event_type")
        details = entry.get("details", {})
        if event == "TOOL_CALL":
            tool_calls += 1
            if details.get("success") is False:
                tool_failures += 1

        msg = entry.get("message", "")
        if "Cycle" in msg and "iteration" in msg:
            m = re.search(r"Cycle (cycle_\d+).* iteration (\d+)", msg)
            if m:
                cycle_name = m.group(1)
                iteration = int(m.group(2))
                cycle_iterations[cycle_name] = max(cycle_iterations.get(cycle_name, 0), iteration)

    return {
        "task_finished": task_finished,
        "error_count": error_count,
        "warning_count": warning_count,
        "tool_call_count": tool_calls,
        "tool_failure_rate": tool_failures / tool_calls if tool_calls else 0.0,
        "cycle_iterations": cycle_iterations,
    }

def find_execution_log(root: Path) -> Optional[Path]:
    for candidate in root.rglob("execution_logs.json"):
        return candidate
    return None


# ----------------------------------------------------------------------
# 3. Benchmark evaluation (with integrated log analysis)
# ----------------------------------------------------------------------

def _resolve_opik_project_name() -> str:
    return os.getenv("OPIK_PROJECT_NAME", "ChatDev")


def _resolve_opik_workspace() -> str | None:
    return os.getenv("OPIK_WORKSPACE") or os.getenv("OPIK_WORKSPACE_NAME")


def _resolve_opik_host() -> str:
    return os.getenv("OPIK_URL_OVERRIDE", DEFAULT_OPIK_HOST).rstrip("/")


def _build_opik_client() -> object | None:
    global opik
    load_dotenv_file()
    def _load_env_from_ancestors(filename: str = ".env") -> Path | None:
        p = Path.cwd()
        visited = set()
        while True:
            if p in visited:
                break
            visited.add(p)
            candidate = p / filename
            if candidate.exists():
                try:
                    for line in candidate.read_text(encoding="utf-8").splitlines():
                        stripped = line.strip()
                        if not stripped or stripped.startswith("#") or "=" not in stripped:
                            continue
                        key, value = stripped.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        os.environ.setdefault(key, value)
                except Exception:
                    pass
                return candidate
            if p.parent == p:
                break
            p = p.parent
        script_dir = Path(__file__).parent
        candidate = script_dir / filename
        if candidate.exists():
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    key, value = stripped.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key, value)
            except Exception:
                pass
            return candidate
        return None
    _load_env_from_ancestors()
    api_key = os.getenv("OPIK_API_KEY")
    if not api_key:
        return None
    if opik is None:
        try:
            opik = import_module("opik")
        except Exception as exc:
            print(f"OPIK_API_KEY found but the Python package 'opik' could not be imported; logging disabled: {exc}", file=sys.stderr)
            return None
    try:
        return opik.Opik(
            project_name=_resolve_opik_project_name(),
            workspace=_resolve_opik_workspace(),
            host=_resolve_opik_host(),
            api_key=api_key,
        )
    except Exception:
        return None


def _iter_text_files(root: Path) -> Iterable[Path]:
    skip_names = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", "benchmark_results"}
    for path in root.rglob("*"):
        if any(part in skip_names for part in path.parts):
            continue
        if path.is_file():
            yield path


def _matches_any(rel_path: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch(rel_path, pattern) for pattern in patterns)


def _should_skip_archive_member(member_path: PurePosixPath) -> bool:
    skip_parts = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", "dist", "build"}
    return any(part in skip_parts for part in member_path.parts)


def _search_pattern_matches(root: Path, pattern: str, paths: Sequence[str] = ()) -> tuple[int, str]:
    regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    matched_files: list[str] = []
    examples: list[str] = []
    match_count = 0
    for file_path in _iter_text_files(root):
        rel = file_path.relative_to(root).as_posix()
        if not _matches_any(rel, paths):
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        file_matches = list(regex.finditer(content))
        if not file_matches:
            continue
        match_count += len(file_matches)
        matched_files.append(rel)
        for match in file_matches:
            if len(examples) >= 5:
                break
            line_number = content.count("\n", 0, match.start()) + 1
            line = content.splitlines()[line_number - 1].strip()
            examples.append(f"{rel}:{line_number}: {line}")
    if match_count:
        suffix = f"; examples: {'; '.join(examples)}" if examples else ""
        return match_count, f"{match_count} match(es) in: {', '.join(matched_files[:5])}{suffix}"
    return 0, f"pattern not found: {pattern}"


def _search_pattern(root: Path, pattern: str, paths: Sequence[str] = ()) -> tuple[bool, str]:
    match_count, details = _search_pattern_matches(root, pattern, paths)
    if match_count:
        matched_part = details.split("; examples:", 1)[0]
        return True, matched_part.replace(f"{match_count} match(es)", "matched")
    return False, f"pattern not found: {pattern}"


def _check_file_exists(root: Path, paths: Sequence[str]) -> tuple[bool, str]:
    found = [candidate for candidate in paths if (root / candidate).exists()]
    if found:
        return True, f"found: {', '.join(found)}"
    return False, f"missing any of: {', '.join(paths)}"


def _run_command(command: Sequence[str], cwd: Path, timeout_seconds: int) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        return False, f"command not available: {exc}"
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout_seconds}s"
    details = (completed.stdout or "") + (completed.stderr or "")
    details = details.strip()
    if completed.returncode == 0:
        return True, details or "exit 0"
    return False, details or f"exit {completed.returncode}"


def _evaluate_check(root: Path, spec: CheckSpec) -> CheckResult:
    started = perf_counter()
    passed = False
    details = ""
    match_count: int | None = None
    if spec.kind == "file_exists":
        passed, details = _check_file_exists(root, spec.paths)
    elif spec.kind == "grep":
        if not spec.pattern:
            raise ValueError(f"missing pattern for {spec.check_id}")
        if spec.must_absent:
            match_count, details = _search_pattern_matches(root, spec.pattern, spec.paths)
            passed = match_count == 0
            details = "0 incomplete stubs found" if passed else details
        else:
            passed, details = _search_pattern(root, spec.pattern, spec.paths)
        if spec.must_absent:
            match_count = 0 if match_count is None else match_count
    elif spec.kind == "command":
        if not spec.command:
            raise ValueError(f"missing command for {spec.check_id}")
        command_root = root if spec.command_cwd is None else (root / spec.command_cwd)
        passed, details = _run_command(spec.command, command_root, spec.timeout_seconds)
    else:
        raise ValueError(f"unsupported check kind: {spec.kind}")
    duration = perf_counter() - started
    return CheckResult(
        check_id=spec.check_id,
        description=spec.description,
        kind=spec.kind,
        passed=passed,
        details=details,
        weight=spec.weight,
        duration_seconds=duration,
        match_count=match_count,
    )


def _score_to_rating(score: float) -> int:
    score = max(0, min(100, score))
    return min(10, int(score // 10) + 1)


def _thread_id_from_target(target: Path) -> str | None:
    name = target.stem if target.suffix else target.name
    for benchmark_name in sorted(BENCHMARK_NAMES, key=len, reverse=True):
        prefix = f"{benchmark_name}_"
        if name.startswith(prefix):
            thread_id = name[len(prefix):]
            return thread_id or None
        if name == benchmark_name:
            return None
    return None


def _extract_archive(archive: Path, temp_root: Path) -> Path:
    target = temp_root / archive.stem
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            member_name = member.filename.replace("\\", "/")
            member_path = PurePosixPath(member_name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if _should_skip_archive_member(member_path):
                continue
            destination = target.joinpath(*member_path.parts)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(member) as source, destination.open("wb") as destination_file:
                destination_file.write(source.read())
    return target


def _locate_project_root(extracted_root: Path) -> Path:
    code_workspaces = [path for path in extracted_root.rglob("code_workspace") if path.is_dir()]
    if code_workspaces:
        return code_workspaces[0]
    candidate_files = ["Dockerfile", "pyproject.toml", "package.json", "requirements.txt", "compose.yml", "docker-compose.yml"]
    for candidate in candidate_files:
        if (extracted_root / candidate).exists():
            return extracted_root
    subdirs = [path for path in extracted_root.iterdir() if path.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    return extracted_root


def _benchmark_name_from_path(path: Path) -> str | None:
    name = path.stem if path.is_file() else path.name
    for benchmark_name in BENCHMARK_NAMES:
        if name.startswith(benchmark_name):
            return benchmark_name
    return None


def _collect_targets(input_path: Path, benchmark: str | None) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        archives = sorted(input_path.glob("*.zip"))
        project_markers = ["Dockerfile", "pyproject.toml", "package.json", "requirements.txt", "compose.yml", "docker-compose.yml", "code_workspace"]
        looks_like_project = any((input_path / marker).exists() for marker in project_markers)
        if looks_like_project and benchmark:
            return [input_path]
        if benchmark:
            selected = [archive for archive in archives if _benchmark_name_from_path(archive) == benchmark]
            return selected
        if looks_like_project and not archives:
            return [input_path]
        return archives
    raise FileNotFoundError(f"input path not found: {input_path}")


def _log_result_to_opik(result: dict, client: object | None) -> bool:
    if client is None:
        return False
    thread_id = result.get("thread_id")
    if not thread_id:
        return False

    completeness_score = float(result.get("completeness_score", result.get("score", 0.0)))
    rating = _score_to_rating(completeness_score)
    reason = (
        f"requirements completeness={completeness_score:.2f}/100; "
        f"passed={result.get('passed_checks', 0)}/{result.get('total_checks', 0)}"
    )
    scores = [{"id": thread_id, "name": OPIK_SCORE_NAME, "value": completeness_score, "reason": reason}]
    if result.get("incomplete_stub_count") is not None:
        incomplete_stub_count = float(result.get("incomplete_stub_count", 0))
        scores.append({
            "id": thread_id,
            "name": OPIK_NO_STUBS_SCORE_NAME,
            "value": incomplete_stub_count,
            "reason": f"{incomplete_stub_count:.0f} incomplete stub(s); {result.get('no_stubs_details', 'placeholder/stub scan')}",
        })

    metrics = result.get("thread_metrics", {})
    if metrics:
        # Only log metrics that actually exist in the returned dict
        existing_keys = ["task_finished", "error_count", "warning_count", "tool_call_count", "tool_failure_rate"]
        for key in existing_keys:
            if key in metrics:
                val = metrics[key]
                if isinstance(val, (int, float, bool)):
                    scores.append({"id": thread_id, "name": f"thread_{key}", "value": float(val), "reason": "execution log analysis"})
        for cycle, iters in metrics.get("cycle_iterations", {}).items():
            scores.append({"id": thread_id, "name": f"thread_{cycle}_iterations", "value": float(iters), "reason": "execution log analysis"})

    try:
        client.log_threads_feedback_scores(scores=scores, project_name=_resolve_opik_project_name())
        return True
    except Exception as exc:
        print(f"Opik logging failed for thread {thread_id}: {exc}", file=sys.stderr)
        return False


def evaluate_target(target: Path, benchmark: str | None = None) -> dict:
    """Evaluate a single target (zip or directory) and return combined results."""
    benchmark_name = benchmark or _benchmark_name_from_path(target)
    if benchmark_name is None:
        raise ValueError(f"Could not infer benchmark name from {target}")

    specs = _benchmark_checks().get(benchmark_name)
    if specs is None:
        raise ValueError(f"No checks registered for {benchmark_name}")

    with tempfile.TemporaryDirectory(prefix="chatdev-benchmark-", ignore_cleanup_errors=True) as temp_dir:
        temp_root = Path(temp_dir)
        if target.is_file() and target.suffix.lower() == ".zip":
            extracted_root = _extract_archive(target, temp_root)
        else:
            extracted_root = target

        project_root = _locate_project_root(extracted_root)

        # ---- Static checks ----
        hidden_spec_text = _load_hidden_spec_text(benchmark_name)
        results: list[CheckResult] = []
        for spec in specs:
            if _should_skip_check_by_hidden_spec(benchmark_name, spec, hidden_spec_text):
                results.append(CheckResult(check_id=spec.check_id, description=spec.description, kind=spec.kind,
                                           passed=True, details="skipped per hidden spec", weight=0.0, duration_seconds=0.0))
            else:
                results.append(_evaluate_check(project_root, spec))

        score_summary = _score_results(results)
        score = score_summary["completeness_score"]
        thread_id = _thread_id_from_target(target)
        rating = score_summary["completeness_rating"]

        # ---- Execution log analysis ----
        thread_metrics = {}
        log_file = find_execution_log(extracted_root)
        if log_file:
            try:
                logs = parse_execution_logs(log_file)
                thread_metrics = analyze_logs(logs)
            except Exception as e:
                print(f"Warning: failed to analyze execution logs for {target}: {e}", file=sys.stderr)
        else:
            print(f"Info: no execution_logs.json found for {target}", file=sys.stderr)

        return {
            "benchmark": benchmark_name,
            "target": str(target),
            "project_root": str(project_root),
            "score": round(score, 2),
            "completeness_score": score_summary["completeness_score"],
            "rating": rating,
            "thread_id": thread_id,
            "passed_checks": score_summary["passed_checks"],
            "total_checks": score_summary["total_checks"],
            "no_stubs_passed": score_summary["no_stubs_passed"],
            "no_stubs_details": score_summary["no_stubs_details"],
            "incomplete_stub_count": score_summary["incomplete_stub_count"],
            "checks": [r.__dict__ for r in results],
            "thread_metrics": thread_metrics,
        }


def evaluate_and_log_target(target: Path, benchmark: str | None = None, opik_client: object | None = None) -> dict:
    result = evaluate_target(target, benchmark=benchmark)
    client = opik_client if opik_client is not None else _build_opik_client()
    result["opik_logged"] = _log_result_to_opik(result, client)
    return result


def _print_report(result: dict) -> None:
    print(f"[{result['benchmark']}] requirements coverage: {result['completeness_score']:.2f}% ({result['passed_checks']}/{result['total_checks']})")
    print(f"  target: {result['target']}")
    print(f"  root:   {result['project_root']}")
    if result.get("thread_id"):
        print(f"  thread: {result['thread_id']}")
    if result.get("no_stubs_passed") is not None:
        count = result.get("incomplete_stub_count")
        status = "PASS" if result["no_stubs_passed"] else "FAIL"
        print(f"  incomplete stubs: {count} ({status}; {result.get('no_stubs_details', 'not evaluated')})")
    for check in result["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  - {status} {check['check_id']}: {check['description']} ({check['details']})")
    tm = result.get("thread_metrics", {})
    if tm:
        print(f"  Thread metrics: finished={tm.get('task_finished')}, errors={tm.get('error_count')}, "
              f"warnings={tm.get('warning_count')}, tool_calls={tm.get('tool_call_count')}, "
              f"tool_failure_rate={tm.get('tool_failure_rate', 0):.2f}, cycles={tm.get('cycle_iterations', {})}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ChatDev benchmark outputs with auto‑checks and thread log analysis")
    parser.add_argument("path", help="A benchmark .zip archive or a directory containing benchmark archives")
    parser.add_argument("--benchmark", choices=BENCHMARK_NAMES, help="Evaluate only the specified benchmark when the input is a directory")
    parser.add_argument("--json-output", help="Optional path to write a JSON summary")
    args = parser.parse_args()

    input_path = Path(args.path).expanduser().resolve()
    targets = _collect_targets(input_path, args.benchmark)
    if not targets:
        print("No benchmark archives found.", file=sys.stderr)
        return 1

    opik_client = _build_opik_client()
    results = [evaluate_and_log_target(t, benchmark=args.benchmark, opik_client=opik_client) for t in targets]
    summary = {"results": results}

    for result in results:
        _print_report(result)

    if args.json_output:
        output_path = Path(args.json_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote JSON summary to {output_path}")

    if opik_client is not None:
        logged_count = sum(1 for r in results if r.get("opik_logged"))
        print(f"\nOpik logging attempted for {logged_count}/{len(results)} result(s)")
    else:
        print("\nOpik logging skipped: set OPIK_API_KEY to enable thread feedback upload.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

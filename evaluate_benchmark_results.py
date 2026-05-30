"""Post-hoc benchmark evaluator for generated ChatDev projects.

The script accepts either a benchmark session archive or a directory that
contains one or more session archives. It extracts each archive, locates the
generated project root, executes a small set of auto-checks, and prints a score
per benchmark.

The checks are intentionally conservative: static greps, file existence checks,
and optional command execution such as ``pytest``. The implementation is
designed to be extended with more benchmark-specific commands later if needed.
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
from fnmatch import fnmatch
from importlib import import_module
from pathlib import Path
from pathlib import PurePosixPath
from time import perf_counter
from typing import Iterable, Sequence

from utils.env_loader import load_dotenv_file


BENCHMARK_NAMES = [
    "benchmark_task_api",
    "benchmark_csv_pipeline",
    "benchmark_chat_server",
    "benchmark_url_shortener",
    "benchmark_expense_tracker",
]

OPIK_SCORE_NAME = "benchmark_requirements_rating"
DEFAULT_OPIK_HOST = "https://www.comet.com/opik/api"

try:
    opik = import_module("opik")
except Exception:
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
    return {
        "benchmark_task_api": [
            CheckSpec("AC1", "Project contains a Dockerfile", "file_exists", paths=("Dockerfile",)),
            CheckSpec("AC2", "FastAPI is used", "grep", pattern=r"from\s+fastapi|import\s+fastapi"),
            CheckSpec("AC3", "SQLite is used", "grep", pattern=r"sqlite"),
            CheckSpec("AC4", "JWT auth is implemented", "grep", pattern=r"jwt|bearer|jose"),
            CheckSpec("AC5", "Soft delete fields appear in the code", "grep", pattern=r"is_deleted|deleted_at"),
            CheckSpec("AC6", "Pagination keywords appear in routes", "grep", pattern=r"skip|limit|page", paths=("**/routes/**", "**/*route*.*", "**/*router*.*")),
            CheckSpec("AC7", "User isolation keywords appear in the code", "grep", pattern=r"user_id|owner_id"),
            CheckSpec("AC8", "Pytest suite is runnable", "command", command=(sys.executable, "-m", "pytest", "-q"), timeout_seconds=600),
        ],
        "benchmark_csv_pipeline": [
            CheckSpec("AC1", "Pipeline source exists", "grep", pattern=r"csv|pandas|argparse"),
            CheckSpec("AC2", "JSON output logic appears in the code", "grep", pattern=r"json\.dump|json\.dumps|orjson"),
            CheckSpec("AC3", "Streaming/chunked reading appears in the code", "grep", pattern=r"chunksize|chunk_size|iterchunks|iterrows"),
            CheckSpec("AC4", "Malformed row handling appears in the code", "grep", pattern=r"skip|malformed|invalid|error"),
            CheckSpec("AC5", "Parallel execution is present", "grep", pattern=r"ProcessPoolExecutor|ThreadPoolExecutor|Pool"),
            CheckSpec("AC6", "CLI support is present", "grep", pattern=r"argparse|click|typer"),
            CheckSpec("AC7", "Logging is present", "grep", pattern=r"logging\.|logger\."),
        ],
        "benchmark_chat_server": [
            CheckSpec("AC1", "Project contains a Dockerfile", "file_exists", paths=("Dockerfile",)),
            CheckSpec("AC2", "WebSocket implementation appears in the code", "grep", pattern=r"WebSocket|websocket|ws://"),
            CheckSpec("AC3", "Persistent storage appears in the code", "grep", pattern=r"INSERT|session\.add|db\.execute"),
            CheckSpec("AC4", "Room support appears in the code", "grep", pattern=r"room|channel|room_id"),
            CheckSpec("AC5", "Message ordering keywords appear in the code", "grep", pattern=r"timestamp|created_at|sequence|ORDER BY"),
            CheckSpec("AC6", "Authentication keywords appear near the WS flow", "grep", pattern=r"token|authenticate|jwt|login"),
            CheckSpec("AC7", "Async code appears in the code", "grep", pattern=r"async\s+def|await|asyncio"),
            CheckSpec("AC8", "Pytest suite is runnable", "command", command=(sys.executable, "-m", "pytest", "-q"), timeout_seconds=600),
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
        ],
        "benchmark_expense_tracker": [
            CheckSpec("AC1", "CLI support is present", "grep", pattern=r"argparse|click|typer"),
            CheckSpec("AC2", "Expense CRUD keywords appear in the code", "grep", pattern=r"add|edit|delete|expense"),
            CheckSpec("AC3", "Currency conversion keywords appear in the code", "grep", pattern=r"currency|exchange|convert"),
            CheckSpec("AC4", "Recurring/monthly keywords appear in the code", "grep", pattern=r"recurr|monthly"),
            CheckSpec("AC5", "Monthly summary keywords appear in the code", "grep", pattern=r"summary|report|category"),
            CheckSpec("AC6", "CSV export keywords appear in the code", "grep", pattern=r"csv|export"),
            CheckSpec("AC7", "SQLite usage appears in the code", "grep", pattern=r"sqlite"),
            CheckSpec("AC8", "Pytest suite is runnable", "command", command=(sys.executable, "-m", "pytest", "-q"), timeout_seconds=600),
        ],
    }


def _load_hidden_spec_text(benchmark_name: str) -> str | None:
    """Load the hidden spec text for a given benchmark from the bundled YAML.

    Returns the spec text (lowercased) or None when not available.
    This uses a best-effort extraction so the script doesn't require PyYAML.
    """
    yaml_path = Path("yaml_instance/ChatDev_simple_benchmarks_v1.yaml")
    if not yaml_path.exists():
        return None
    try:
        text = yaml_path.read_text(encoding="utf-8")
    except Exception:
        return None

    # Find the block that starts with "benchmark_name: |-" and capture the indented block
    pattern = re.compile(rf"^{re.escape(benchmark_name)}:\s*\|-\n((?:\s{{4}}.*\n)+)", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return None
    block = m.group(1)
    # Remove common 4-space indentation
    cleaned = "\n".join(line[4:] if line.startswith("    ") else line for line in block.splitlines())
    return cleaned.strip().lower()


def _should_skip_check_by_hidden_spec(benchmark_name: str, check: CheckSpec, hidden_spec_text: str | None) -> bool:
    """Return True if the given check should be skipped based on the hidden spec.

    We use a small heuristic map: if the hidden spec does not mention keywords
    required by a check, we consider the check inapplicable and skip it.
    """
    if not hidden_spec_text:
        return False

    # Map of benchmark -> check_id -> list of keywords that must appear in hidden spec
    conditional_requirements: dict[str, dict[str, list[str]]] = {
        "benchmark_chat_server": {
            "AC3": ["persist", "database", "sqlite", "store", "history"],
        },
        "benchmark_csv_pipeline": {
            "AC5": ["parallel", "parallel processing", "processpoolexecutor", "threadpoolexecutor"],
            "AC3": ["stream", "streaming", "chunksize", "chunk"],
        },
        "benchmark_task_api": {
            "AC3": ["sqlite", "database", "persist"],
            "AC5": ["soft delete", "is_deleted", "deleted_at"],
            "AC6": ["pagination", "skip", "limit", "page"],
        },
        "benchmark_url_shortener": {
            "AC8": ["postgres", "postgresql", "psycopg"],
        },
        "benchmark_expense_tracker": {
            "AC8": ["pytest", "test suite", "tests"],
        },
    }

    bm_map = conditional_requirements.get(benchmark_name)
    if not bm_map:
        return False

    reqs = bm_map.get(check.check_id)
    if not reqs:
        return False

    # If any of the required keywords is present, the check is applicable.
    for kw in reqs:
        if kw.lower() in hidden_spec_text:
            return False

    # No keyword matched -> skip the check
    return True


def _resolve_opik_project_name() -> str:
    return os.getenv("OPIK_PROJECT_NAME", "ChatDev")


def _resolve_opik_workspace() -> str | None:
    return os.getenv("OPIK_WORKSPACE") or os.getenv("OPIK_WORKSPACE_NAME")


def _resolve_opik_host() -> str:
    return os.getenv("OPIK_URL_OVERRIDE", DEFAULT_OPIK_HOST).rstrip("/")


def _build_opik_client() -> object | None:
    # Load .env using the shared helper, then fall back to scanning parent
    # directories for a .env file in case the process cwd is different.
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
        # Try script directory as a last resort
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

    if opik is None:
        # If an API key exists but the 'opik' package is missing, provide a clearer message.
        if os.getenv("OPIK_API_KEY"):
            print(
                "OPIK_API_KEY found but the Python package 'opik' is not installed;"
                " set up the package to enable logging.",
                file=sys.stderr,
            )
        return None

    api_key = os.getenv("OPIK_API_KEY")
    if not api_key:
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
    skip_parts = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "dist",
        "build",
    }
    return any(part in skip_parts for part in member_path.parts)


def _search_pattern(root: Path, pattern: str, paths: Sequence[str] = ()) -> tuple[bool, str]:
    regex = re.compile(pattern, re.IGNORECASE)
    matches: list[str] = []
    for file_path in _iter_text_files(root):
        rel = file_path.relative_to(root).as_posix()
        if not _matches_any(rel, paths):
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if regex.search(content):
            matches.append(rel)
            if len(matches) >= 5:
                break
    if matches:
        return True, f"matched in: {', '.join(matches)}"
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

    if spec.kind == "file_exists":
        passed, details = _check_file_exists(root, spec.paths)
    elif spec.kind == "grep":
        if not spec.pattern:
            raise ValueError(f"missing pattern for {spec.check_id}")
        passed, details = _search_pattern(root, spec.pattern, spec.paths)
        if spec.must_absent:
            passed = not passed
            details = "pattern absent" if passed else details
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
    )


def _score_to_rating(score: float) -> int:
    score = max(0, min(100, score))
    return min(10, int(score // 10) + 1)


def _thread_id_from_target(target: Path) -> str | None:
    name = target.stem if target.suffix else target.name
    for benchmark_name in sorted(BENCHMARK_NAMES, key=len, reverse=True):
        prefix = f"{benchmark_name}_"
        if name.startswith(prefix):
            thread_id = name[len(prefix) :]
            return thread_id or None
        if name == benchmark_name:
            return None
    return None


def _log_result_to_opik(result: dict, client: object | None) -> bool:
    if client is None:
        return False

    thread_id = result.get("thread_id")
    if not thread_id:
        return False

    rating = _score_to_rating(float(result.get("score", 0.0)))
    reason = (
        f"auto-check score={result.get('score', 0.0):.2f}/100; "
        f"passed={result.get('passed_checks', 0)}/{result.get('total_checks', 0)}; "
        f"rating={rating}/10"
    )
    payload = [
        {
            "id": thread_id,
            "name": OPIK_SCORE_NAME,
            "value": rating,
            "reason": reason,
        }
    ]

    try:
        client.log_threads_feedback_scores(
            scores=payload,
            project_name=_resolve_opik_project_name(),
        )
        return True
    except Exception as exc:
        print(f"Opik logging failed for thread {thread_id}: {exc}", file=sys.stderr)
        return False


def _locate_project_root(extracted_root: Path) -> Path:
    code_workspaces = [path for path in extracted_root.rglob("code_workspace") if path.is_dir()]
    if code_workspaces:
        return code_workspaces[0]

    candidate_files = [
        "Dockerfile",
        "pyproject.toml",
        "package.json",
        "requirements.txt",
        "compose.yml",
        "docker-compose.yml",
    ]
    for candidate in candidate_files:
        if (extracted_root / candidate).exists():
            return extracted_root

    subdirs = [path for path in extracted_root.iterdir() if path.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]

    return extracted_root


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
        project_markers = [
            "Dockerfile",
            "pyproject.toml",
            "package.json",
            "requirements.txt",
            "compose.yml",
            "docker-compose.yml",
            "code_workspace",
        ]
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


def evaluate_target(target: Path, benchmark: str | None = None) -> dict:
    benchmark_name = benchmark or _benchmark_name_from_path(target)
    if benchmark_name is None:
        raise ValueError(f"Could not infer benchmark name from {target}")

    specs = _benchmark_checks().get(benchmark_name)
    if specs is None:
        raise ValueError(f"No checks registered for {benchmark_name}")

    with tempfile.TemporaryDirectory(prefix="chatdev-benchmark-") as temp_dir:
        temp_root = Path(temp_dir)
        if target.is_file() and target.suffix.lower() == ".zip":
            extracted_root = _extract_archive(target, temp_root)
        else:
            extracted_root = target

        project_root = _locate_project_root(extracted_root)
        # Load the hidden spec for this benchmark (if available) and skip checks
        # that are clearly inapplicable according to the hidden specification.
        hidden_spec_text = _load_hidden_spec_text(benchmark_name)
        results: list[CheckResult] = []
        for spec in specs:
            if _should_skip_check_by_hidden_spec(benchmark_name, spec, hidden_spec_text):
                # mark skipped checks with zero weight so they don't affect totals
                results.append(
                    CheckResult(
                        check_id=spec.check_id,
                        description=spec.description,
                        kind=spec.kind,
                        passed=True,
                        details="skipped per hidden spec",
                        weight=0.0,
                        duration_seconds=0.0,
                    )
                )
            else:
                results.append(_evaluate_check(project_root, spec))

        total_weight = sum(result.weight for result in results)
        passed_weight = sum(result.weight for result in results if result.passed)
        score = 100.0 * passed_weight / total_weight if total_weight else 0.0
        thread_id = _thread_id_from_target(target)
        rating = _score_to_rating(score)

        return {
            "benchmark": benchmark_name,
            "target": str(target),
            "project_root": str(project_root),
            "score": round(score, 2),
            "rating": rating,
            "thread_id": thread_id,
            "passed_checks": sum(1 for result in results if result.passed),
            "total_checks": len(results),
            "checks": [result.__dict__ for result in results],
        }


def evaluate_and_log_target(target: Path, benchmark: str | None = None, opik_client: object | None = None) -> dict:
    result = evaluate_target(target, benchmark=benchmark)
    client = opik_client if opik_client is not None else _build_opik_client()
    result["opik_logged"] = _log_result_to_opik(result, client)
    return result


def _print_report(result: dict) -> None:
    print(
        f"[{result['benchmark']}] score: {result['score']:.2f} "
        f"rating: {result['rating']}/10 ({result['passed_checks']}/{result['total_checks']})"
    )
    print(f"  target: {result['target']}")
    print(f"  root:   {result['project_root']}")
    if result.get("thread_id"):
        print(f"  thread: {result['thread_id']}")
    for check in result["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  - {status} {check['check_id']}: {check['description']} ({check['details']})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ChatDev benchmark outputs with auto-checks")
    parser.add_argument(
        "path",
        help="A benchmark .zip archive or a directory containing benchmark archives",
    )
    parser.add_argument(
        "--benchmark",
        choices=BENCHMARK_NAMES,
        help="Evaluate only the specified benchmark when the input is a directory",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path to write a JSON summary",
    )
    args = parser.parse_args()

    input_path = Path(args.path).expanduser().resolve()
    targets = _collect_targets(input_path, args.benchmark)
    if not targets:
        print("No benchmark archives found.", file=sys.stderr)
        return 1

    opik_client = _build_opik_client()
    results = [evaluate_and_log_target(target, benchmark=args.benchmark, opik_client=opik_client) for target in targets]
    summary = {"results": results}

    for result in results:
        _print_report(result)

    if args.json_output:
        output_path = Path(args.json_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote JSON summary to {output_path}")

    if opik_client is not None:
        logged_count = sum(1 for result in results if result.get("opik_logged"))
        print(f"\nOpik logging attempted for {logged_count}/{len(results)} result(s)")
    else:
        print("\nOpik logging skipped: set OPIK_API_KEY to enable thread feedback upload.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
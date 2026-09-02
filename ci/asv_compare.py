"""Benchmark two or more git revisions of versioned-hdf5 and ``asv compare`` each of
them against the first one.

Each revision is built from a temporary git worktree and installed, one at a time, in
the current environment; the working tree is installed back at the end. benchmarks/ and
asv.conf.json always come from the working tree, never from the revisions, so
benchmarks/ must be able to run on all of them.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / ".asv" / "results"  # results_dir in asv.conf.json

# ``asv run`` exits with this code when at least one benchmark failed. It's not fatal
# here; the report marks the failures.
ASV_BENCHMARK_FAILED = 2


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pixi r asv-compare",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "revs",
        nargs="+",
        metavar="REV",
        help="Two or more git revisions to benchmark: a branch, a tag, or a hash",
    )
    parser.add_argument(
        "-n",
        "--repeat",
        type=int,
        default=3,
        help="How many samples of each benchmark to take (default: 3)",
    )
    parser.add_argument(
        "-b",
        "--bench",
        action="append",
        default=[],
        metavar="REGEX",
        help="Only run the benchmarks matching this regex. May be repeated. "
        "Forwarded to `asv run`.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=RESULTS_DIR.parent / "compare_report.md",
        metavar="FILE",
        help="Path to write the comparison report (default: .asv/compare_report.md)",
    )
    args = parser.parse_args(argv)
    if len(set(args.revs)) < 2:  # Deduplicate
        parser.error("need at least two revisions to compare")
    return args


def run(*args: str | Path, tee: io.TextIOBase | None = None) -> None:
    """Run a command in the project root, raising if it fails.
    If `tee` is given, stdout is also written to that file.
    """
    print("+", *args, flush=True)
    rbytes = subprocess.check_output(args, cwd=PROJECT_ROOT)
    result = rbytes.decode("utf-8", errors="ignore")
    sys.stdout.write(result)
    sys.stdout.flush()
    if tee is not None:
        tee.write(result)


def asv_run(commit_hash: str, repeat: int, bench: list[str]) -> None:
    """Benchmark whatever is installed in the current environment and store the results
    under commit_hash
    """
    args: list[str] = ["--bench=" + regex for regex in bench]
    try:
        run(
            sys.executable,
            "-m",
            "asv",
            "run",
            "--python=same",
            f"--attribute=repeat={repeat}",
            # Benchmarks that alter the state of what they measure need setup() to run
            # again before every call, and setup() doesn't run between the `number`
            # calls of a sample nor during warmup.
            "--attribute=number=1",
            "--attribute=warmup_time=0",
            "--set-commit-hash",
            commit_hash,
            *args,
        )
    except subprocess.CalledProcessError as e:
        if e.returncode != ASV_BENCHMARK_FAILED:
            raise RuntimeError(
                f"`asv run` failed with exit code {e.returncode}. If ASV was never "
                "initialised on this machine, run `pixi r asv-machine` first."
            ) from None


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    revs = {}
    for rev in args.revs:
        try:
            revs[rev] = subprocess.check_output(
                ["git", "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"],
                cwd=PROJECT_ROOT,
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            raise ValueError(f"Not a git revision: {rev}") from None

    # Don't accidentally pick up old results
    for commit_hash in revs.values():
        for path in RESULTS_DIR.glob(f"*/{commit_hash[:8]}-*.json"):
            print(f"Removing stale {path}")
            path.unlink()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for rev, commit_hash in revs.items():
                print(f"=== {rev}", flush=True)
                worktree = f"{tmpdir}/{commit_hash}"
                run("git", "worktree", "add", "--detach", worktree, commit_hash)
                run(
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    "--no-build-isolation",
                    "--no-dependencies",
                    worktree,
                )
                asv_run(commit_hash, args.repeat, args.bench)
    finally:
        run(sys.executable, PROJECT_ROOT / "ci" / "editable_install.py", "--force")
        # The worktrees are gone along with tmpdir, but git still remembers them
        run("git", "worktree", "prune")

    baseline, *others = revs

    with open(args.output, "w") as f:
        for rev in others:
            run(sys.executable, "-m", "asv", "compare", baseline, rev, tee=f)

    print(f"Report written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""Build the Kaggle upload ZIP and the repository pointer file.

The competition documentation contradicts itself about the ZIP contents:

* the Rules page says a submission must be "a ZIP containing exactly
  submission.csv" and that it "must not include submission_metadata.json,
  repo_link.txt, or any other repository metadata file";
* the later "Final submission format" forum thread says the ZIP must contain
  "exactly two files: submission.csv and repo_link.txt".

The Rules page states that the foundational and competition rules control, so
the default here follows it and packages `submission.csv` alone — which is what
the scored submission used. `--include-repo-link` produces the forum-thread
variant if organizers confirm that reading.

`repo_link.txt` is written to the repository root either way, since the thread
requires its exact two-line form: the repository URL, then the full 40-character
commit hash.
"""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = ROOT / "artifacts" / "submission"
SAMPLE = ROOT / "sample_submission.csv"


def current_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    commit = result.stdout.strip()
    return commit if len(commit) == 40 else None


def verify(csv_path: Path) -> pd.DataFrame:
    """Re-check the submission against the published row ids before packaging."""
    submission = pd.read_csv(csv_path)
    if list(submission.columns) != ["row_id", "failure_probability"]:
        raise SystemExit(f"columns must be row_id,failure_probability; got {list(submission.columns)}")
    if submission["row_id"].duplicated().any():
        raise SystemExit("duplicate row_id values")
    probabilities = submission["failure_probability"]
    if probabilities.isna().any():
        raise SystemExit("failure_probability contains nulls")
    if not probabilities.between(0.0, 1.0).all():
        raise SystemExit("failure_probability must lie in [0, 1]")
    if SAMPLE.exists():
        required = pd.read_csv(SAMPLE, usecols=["row_id"])
        if not submission["row_id"].equals(required["row_id"]):
            raise SystemExit("row_id values must match sample_submission.csv exactly, in order")
        print(f"verified against {SAMPLE.name}: {len(submission):,} rows, order matches")
    else:
        print(f"{SAMPLE.name} not present; checked format only ({len(submission):,} rows)")
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the Kaggle submission ZIP.")
    parser.add_argument("--repo-url", help="https://github.com/USER/REPO")
    parser.add_argument("--commit", help="40-character commit hash; read from git when omitted")
    parser.add_argument(
        "--include-repo-link",
        action="store_true",
        help="Put repo_link.txt inside the ZIP (forum-thread reading; the Rules page forbids it)",
    )
    args = parser.parse_args()

    csv_path = SUBMISSION / "submission.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found; run the pipeline first")
    verify(csv_path)

    commit = args.commit or current_commit()
    if args.repo_url and commit:
        pointer = ROOT / "repo_link.txt"
        pointer.write_text(f"{args.repo_url}\n{commit}\n", encoding="utf-8")
        print(f"wrote {pointer.name}: {args.repo_url} @ {commit}")
    elif args.include_repo_link:
        raise SystemExit("--include-repo-link needs --repo-url and a resolvable commit hash")

    zip_path = SUBMISSION / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="submission.csv")
        if args.include_repo_link:
            archive.write(ROOT / "repo_link.txt", arcname="repo_link.txt")

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    expected = ["submission.csv", "repo_link.txt"] if args.include_repo_link else ["submission.csv"]
    if sorted(names) != sorted(expected):
        raise SystemExit(f"ZIP contains {names}, expected {expected}")
    print(f"wrote {zip_path} containing {names}")


if __name__ == "__main__":
    main()

"""Documentation contracts; no video, model, or historical evidence is opened."""

from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import subprocess
import sys
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
PHASE17_DOCS = (
    "ARCHITECTURE.md", "STATE_MACHINE.md", "SQLITE.md", "ML_EVALUATION.md",
    "PERFORMANCE.md", "TESTING.md", "LIMITATIONS.md", "CLAIMS.md",
    "REPRODUCIBILITY.md", "INTERVIEW_GUIDE.md", "ENGINEERING_SUMMARY.md",
    "PHASE17_DOCUMENTATION.md",
)
LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")


def test_phase17_documents_and_relative_links_resolve():
    documents = [ROOT / "README.md", *(ROOT / "docs" / name for name in PHASE17_DOCS)]
    assert all(path.is_file() for path in documents)
    broken = []
    for document in documents:
        for target in LINK.findall(document.read_text(encoding="utf-8")):
            destination = unquote(target.split("#", 1)[0].strip("<>"))
            if not destination or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", destination):
                continue
            if not (document.parent / destination).exists():
                broken.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not broken, "Broken relative documentation links:\n" + "\n".join(broken)


def test_cli_examples_parse_against_the_actual_parser(monkeypatch, capsys):
    import main

    @dataclass
    class Captured:
        options: dict

    monkeypatch.setattr(main, "run", lambda **options: Captured(options))
    examples = []
    for path in (ROOT / "README.md", ROOT / "docs" / "SQLITE.md"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("python main.py "):
                examples.append(line)
    assert len(examples) >= 7
    for line in examples:
        argv = shlex.split(line, posix=True)
        assert argv[:2] == ["python", "main.py"]
        assert main.main(argv[2:]) == 0
        capsys.readouterr()
    for path in (ROOT / "main.py", ROOT / "tools" / "query_sessions.py"):
        help_run = subprocess.run(
            [sys.executable, "-B", str(path), "--help"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        )
        assert "usage:" in help_run.stdout.lower()


def test_claims_keep_baseline_and_candidate_distinct():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    evaluation = (ROOT / "docs" / "ML_EVALUATION.md").read_text(encoding="utf-8")
    claims = (ROOT / "docs" / "CLAIMS.md").read_text(encoding="utf-8")
    for text in (readme, evaluation, claims):
        assert "Phase 11 CNN-LSTM16" in text or "CNN-LSTM16 baseline" in text
        assert "development cross-validation" in text.lower()
        assert "no pristine external final-test estimate" in text.lower()
    assert "94.1909%" in evaluation and "0.949829" in evaluation
    assert "final LSTM32 model achieved 94.19%" in claims  # Explicitly forbidden example.

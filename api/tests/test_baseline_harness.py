"""
Reuses the existing harness scripts via their main() to assert exit codes (plan: "Reuse
run_program.py/validate_run_doc.py via their main()"). This pins the committed hero
cache/program/run-doc as the baseline the pipeline must reproduce.

These scripts read sys.argv directly (argparse / sys.argv[1]), so each test pins argv to
just the program name — otherwise pytest's own argv leaks in.
"""
from __future__ import annotations

import run_program
import validate_run_doc
import validate_program


def test_run_program_main_exits_zero_on_committed_hero(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_program.py"])
    rc = run_program.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "Yes — #10 scored the goal" in out
    assert "run-doc valid" in out


def test_validate_run_doc_main_exits_zero(monkeypatch):
    monkeypatch.setattr("sys.argv", ["validate_run_doc.py"])
    assert validate_run_doc.main() == 0


def test_validate_program_main_exits_zero(monkeypatch):
    monkeypatch.setattr("sys.argv", ["validate_program.py"])
    assert validate_program.main() == 0

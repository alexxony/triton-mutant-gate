from pathlib import Path

from mutant_gate.cli import main

VECADD = str(Path(__file__).parent.parent / "examples" / "kernels" / "vecadd.py")


def test_mutants_lists_operators(capsys):
    assert main(["mutants", VECADD]) == 0
    out = capsys.readouterr()
    assert "drop_load_mask" in out.out and "4 mutants" in out.err


def test_mutants_diff_shows_the_change(capsys):
    assert main(["mutants", VECADD, "--diff"]) == 0
    assert "mask_lt_to_le" in capsys.readouterr().out


def test_bad_script_is_a_clean_error(tmp_path, capsys):
    bad = tmp_path / "k.py"
    bad.write_text("print('x')\n")
    assert main(["mutants", str(bad)]) == 1
    assert "error:" in capsys.readouterr().err


def test_missing_file_is_a_clean_error(capsys):
    assert main(["mutants", "/nonexistent.py"]) == 1

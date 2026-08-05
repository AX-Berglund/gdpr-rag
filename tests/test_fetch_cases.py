"""Tests for case-reference handling."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_cases import read_list, to_celex  # noqa: E402


class TestCaseReferences:
    @pytest.mark.parametrize(
        "reference,celex",
        [
            ("C-300/21", "62021CJ0300"),
            ("C-154/21", "62021CJ0154"),
            ("C-40/17", "62017CJ0040"),
            ("c-300/21", "62021CJ0300"),
            ("  C-300/21  ", "62021CJ0300"),
        ],
    )
    def test_case_numbers_convert(self, reference, celex):
        assert to_celex(reference) == celex

    def test_celex_ids_pass_through(self):
        assert to_celex("62021CJ0300") == "62021CJ0300"

    def test_the_year_is_when_the_case_was_lodged(self):
        # C-300/21 was decided in 2023 but lodged in 2021; CELEX uses lodging.
        assert to_celex("C-300/21").startswith("62021")

    @pytest.mark.parametrize("bad", ["300/21", "C300/21", "Case C-300/21", "", "C-300"])
    def test_unrecognised_references_are_rejected(self, bad):
        with pytest.raises(ValueError, match="unrecognised case reference"):
            to_celex(bad)


class TestListParsing:
    def test_comments_and_blanks_are_ignored(self, tmp_path):
        path = tmp_path / "list.txt"
        path.write_text("# a comment\n\nC-300/21\nC-154/21  # trailing note\n")
        assert read_list(path) == ["C-300/21", "C-154/21"]

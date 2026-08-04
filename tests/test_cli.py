"""Tests for the command line interface.

These also guard the packaging entry point: `gdpr-rag = "gdpr_rag.cli:main"`
is declared in pyproject.toml, so `main` must exist and be callable.
"""

import pytest

from gdpr_rag.cli import main


class TestEntryPoint:
    def test_main_is_importable_and_callable(self):
        # The declared console script would crash on install without this.
        assert callable(main)

    def test_no_command_exits_with_usage(self):
        with pytest.raises(SystemExit) as exit_info:
            main([])
        assert exit_info.value.code != 0


class TestIndex:
    def test_missing_corpus_reports_cleanly(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        assert main(["--index", str(tmp_path / "i.sqlite"), "index"]) == 1
        assert "No corpus found" in capsys.readouterr().err

    def test_builds_an_index_from_html(self, tmp_path, capsys):
        from tests.test_eurlex import FIXTURE

        source = tmp_path / "doc.html"
        source.write_text(FIXTURE, encoding="utf-8")
        index = tmp_path / "index.sqlite"

        assert main(["--index", str(index), "index", str(source)]) == 0
        assert index.exists()
        assert "across 3 articles" in capsys.readouterr().out

    def test_reindexing_replaces_rather_than_appends(self, tmp_path, capsys):
        from tests.test_eurlex import FIXTURE

        source = tmp_path / "doc.html"
        source.write_text(FIXTURE, encoding="utf-8")
        index = tmp_path / "index.sqlite"

        main(["--index", str(index), "index", str(source)])
        capsys.readouterr()
        main(["--index", str(index), "index", str(source)])
        out = capsys.readouterr().out
        # A second run must not double the chunk count.
        assert "Indexed 7 chunks" in out


class TestAsk:
    @pytest.fixture
    def index(self, tmp_path):
        from tests.test_eurlex import FIXTURE

        source = tmp_path / "doc.html"
        source.write_text(FIXTURE, encoding="utf-8")
        path = tmp_path / "index.sqlite"
        main(["--index", str(path), "index", str(source)])
        return path

    def test_missing_index_reports_cleanly(self, tmp_path, capsys):
        assert main(["--index", str(tmp_path / "absent.sqlite"), "ask", "anything"]) == 1
        assert "Run 'gdpr-rag index' first" in capsys.readouterr().err

    def test_retrieves_and_prints_citations(self, index, capsys):
        assert main(["--index", str(index), "ask", "erasure of personal data", "-k", "2"]) == 0
        out = capsys.readouterr().out
        assert "Article 17" in out
        assert out.count("[") >= 2

    def test_k_limits_results(self, index, capsys):
        main(["--index", str(index), "ask", "personal data", "-k", "1"])
        out = capsys.readouterr().out
        assert len([ln for ln in out.splitlines() if ln.startswith("[")]) == 1

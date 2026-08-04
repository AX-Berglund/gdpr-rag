"""Tests for local .env loading."""

from gdpr_rag.config import load_env


class TestLoadEnv:
    def test_missing_file_is_not_an_error(self, tmp_path):
        assert load_env(tmp_path / "absent") is False

    def test_values_are_loaded(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEMO_KEY", raising=False)
        path = tmp_path / ".env"
        path.write_text("DEMO_KEY=abc123\n")
        assert load_env(path) is True
        import os

        assert os.environ["DEMO_KEY"] == "abc123"

    def test_exported_variables_win_over_the_file(self, tmp_path, monkeypatch):
        # A stale file must never override a key the user exported deliberately.
        monkeypatch.setenv("DEMO_KEY", "from-shell")
        path = tmp_path / ".env"
        path.write_text("DEMO_KEY=from-file\n")
        load_env(path)
        import os

        assert os.environ["DEMO_KEY"] == "from-shell"

    def test_comments_blanks_and_quotes_are_handled(self, tmp_path, monkeypatch):
        monkeypatch.delenv("QUOTED", raising=False)
        path = tmp_path / ".env"
        path.write_text('# a comment\n\nQUOTED="value"\nnot-a-pair\n')
        load_env(path)
        import os

        assert os.environ["QUOTED"] == "value"

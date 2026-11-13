"""Uploads are private and short-lived: what is kept, what is deleted, and when."""

from pathlib import Path

import pymupdf
import pytest

import rag
import workspace
from rag import DocumentRAG, PageLimitExceeded
from workspace import Workspaces


def make_pdf(path: Path, pages: int) -> Path:
    document = pymupdf.open()
    for number in range(pages):
        document.new_page().insert_text((72, 72), f"page {number + 1} of {path.name}")
    document.save(path)
    document.close()
    return path


# ---- the private folders ----


def test_each_session_gets_its_own_folder(tmp_path):
    spaces = Workspaces(tmp_path, pid=111)
    assert spaces.folder_for("a") != spaces.folder_for("b")
    assert (
        spaces.folder_for("a").is_dir() and spaces.folder_for("a").parent.name == "111"
    )


def test_a_new_session_removes_the_folders_of_sessions_that_are_gone(tmp_path):
    spaces = Workspaces(tmp_path, pid=111)
    for session in ("old-tab", "reloaded-tab", "open-tab", "this-one"):
        (spaces.folder_for(session) / "index.txt").write_text("a user's document text")
    removed = spaces.remove_disconnected({"open-tab", "this-one"})
    assert sorted(removed) == ["old-tab", "reloaded-tab"]
    assert not (tmp_path / "111" / "old-tab").exists()
    assert (
        tmp_path / "111" / "open-tab" / "index.txt"
    ).exists()  # another user is still connected
    assert (tmp_path / "111" / "this-one").exists()


def test_nothing_is_removed_on_a_guess_when_the_session_list_is_unavailable(tmp_path):
    spaces = Workspaces(tmp_path, pid=111)
    spaces.folder_for("a")
    assert spaces.remove_disconnected(None) == []
    assert (tmp_path / "111" / "a").exists()


def test_a_restart_removes_what_an_earlier_server_process_left_behind(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(workspace, "process_is_running", lambda pid: pid == 222)
    for name in ("999", "222", "junk"):
        (tmp_path / name / "session").mkdir(parents=True)
    spaces = Workspaces(tmp_path, pid=111)
    spaces.folder_for("mine")
    spaces.remove_disconnected({"mine"})
    assert not (tmp_path / "999").exists()  # a dead process: its uploads go
    assert not (tmp_path / "junk").exists()
    assert (
        tmp_path / "222"
    ).exists()  # another server that is still running is left alone
    assert (tmp_path / "111" / "mine").exists()


# ---- the page limit ----


def test_the_pages_of_all_pdfs_count_together(tmp_path):
    pdfs = [
        make_pdf(tmp_path / "a.pdf", 12),
        make_pdf(tmp_path / "b.pdf", 9),
    ]  # 21 pages
    with pytest.raises(PageLimitExceeded, match="21 pages in total; the limit is 20"):
        DocumentRAG(pdfs, settings={"max_pages": 20})


def test_the_limit_is_checked_before_any_model_is_used(tmp_path, monkeypatch):
    def not_allowed(*args, **kwargs):
        raise AssertionError("an index was opened for PDFs that are over the limit")

    monkeypatch.setattr(rag, "VectorStore", not_allowed)
    with pytest.raises(PageLimitExceeded):
        DocumentRAG([make_pdf(tmp_path / "a.pdf", 21)], settings={"max_pages": 20})


def test_check_page_limit_allows_exactly_the_limit_and_zero_means_no_limit():
    DocumentRAG.check_page_limit([10, 10], 20)  # exactly 20 is fine
    DocumentRAG.check_page_limit([500], 0)
    DocumentRAG.check_page_limit([500], None)
    with pytest.raises(PageLimitExceeded):
        DocumentRAG.check_page_limit([10, 11], 20)


def test_page_count_reads_the_pages_of_a_pdf(tmp_path):
    assert DocumentRAG.page_count(make_pdf(tmp_path / "a.pdf", 7).read_bytes()) == 7


# ---- what is deleted after indexing ----


class FakeStore:
    """Stands in for the real index: it must be built from PDFs that exist, then they can go."""

    def __init__(self, config):
        self.config, self.built = config, False

    def count(self):
        return 1 if self.built else 0

    def build(self):
        assert list(Path(self.config["data_dir"]).glob("*.pdf")), (
            "the PDFs must exist while indexing"
        )
        self.built = True


class FakeGenerator:
    def __init__(self, config):
        self.cache = None


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "VectorStore", FakeStore)
    monkeypatch.setattr(rag, "Generator", FakeGenerator)
    monkeypatch.setattr(rag, "DOCS_DIR", tmp_path / "vs_docs")


def test_the_uploaded_pdfs_are_deleted_as_soon_as_they_are_indexed(tmp_path, fakes):
    pdf = make_pdf(tmp_path / "a.pdf", 3)
    private = tmp_path / "workspace"
    answerer = DocumentRAG([pdf], workspace=private)
    assert not list(
        private.rglob("*.pdf")
    )  # no copy of the upload is left on the server
    assert answerer.is_available()


def test_a_private_index_is_deleted_when_the_documents_are_replaced(tmp_path, fakes):
    private = tmp_path / "workspace"
    answerer = DocumentRAG([make_pdf(tmp_path / "a.pdf", 3)], workspace=private)
    folder = answerer.folder
    assert folder.exists()
    answerer.close()
    assert not folder.exists()
    assert not answerer.is_available()  # the app then asks the user to upload again


def test_the_command_line_also_keeps_no_pdf_copies(tmp_path, fakes):
    DocumentRAG([make_pdf(tmp_path / "a.pdf", 3)])
    assert not list((tmp_path / "vs_docs").rglob("*.pdf"))


def test_the_answer_cache_lives_in_the_private_folder_and_goes_with_it(tmp_path, fakes):
    private = tmp_path / "workspace"
    answerer = DocumentRAG([make_pdf(tmp_path / "a.pdf", 3)], workspace=private)
    assert Path(answerer.config["cache_file"]).is_relative_to(answerer.folder)
    answerer.close()
    assert not answerer.folder.exists()  # the cache folder was inside it


def test_without_a_private_folder_the_cache_keeps_its_own_temp_folder(tmp_path, fakes):
    assert "cache_file" not in DocumentRAG([make_pdf(tmp_path / "a.pdf", 3)]).config

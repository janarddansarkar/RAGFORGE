"""The web page shows the page count, and will not ingest more than the limit."""

from pathlib import Path

import pymupdf
import pytest
from streamlit.testing.v1 import AppTest

import workspace


@pytest.fixture(autouse=True)
def private_folders_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(
        workspace, "ROOT", tmp_path
    )  # the page's private folders stay in the test's folder


APP = str(Path(__file__).resolve().parent.parent / "src" / "app.py")


def pdf_bytes(pages: int) -> bytes:
    document = pymupdf.open()
    for number in range(pages):
        document.new_page().insert_text((72, 72), f"page {number + 1}")
    data = document.tobytes()
    document.close()
    return data


def page_with(*page_counts: int) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    for field in at.sidebar.text_input:
        field.set_value("sk-test")
    at.run()
    uploader = at.file_uploader[0]
    for index, pages in enumerate(page_counts):
        uploader = uploader.upload(
            f"doc{index}.pdf", pdf_bytes(pages), "application/pdf"
        )
    at.run()
    return at


def ingest_buttons(at: AppTest) -> list:
    return [b for b in at.button if b.label == "Ingest"]


def test_more_than_20_pages_in_total_cannot_be_ingested():
    at = page_with(12, 9)  # 21 pages
    assert any("21 of 20 pages" in e.value for e in at.error)
    assert ingest_buttons(at) == []


def test_exactly_20_pages_can_be_ingested():
    at = page_with(10, 10)
    assert any("20 of 20 pages" in c.value for c in at.caption)
    assert len(ingest_buttons(at)) == 1
    assert not at.error


def test_an_unreadable_pdf_is_reported_and_cannot_be_ingested():
    at = AppTest.from_file(APP, default_timeout=60).run()
    for field in at.sidebar.text_input:
        field.set_value("sk-test")
    at.run()
    at.file_uploader[0].upload("broken.pdf", b"this is not a pdf", "application/pdf")
    at.run()
    assert any("broken.pdf" in e.value for e in at.error)
    assert ingest_buttons(at) == []

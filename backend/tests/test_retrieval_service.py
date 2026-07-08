"""Unit tests for retrieval helpers."""

from app.retrieval.service import image_path_to_url, parse_search_hit


def test_image_path_to_url_from_data_images_path():
    url = image_path_to_url("data/images/abc123/page12_img0.png")
    assert url == "/images/abc123/page12_img0.png"


def test_image_path_to_url_relative_doc_path():
    url = image_path_to_url("abc123/page12_img0.png")
    assert url == "/images/abc123/page12_img0.png"


def test_image_path_to_url_windows_style_path():
    url = image_path_to_url(r"data\images\abc123\page12_img0.png")
    assert url == "/images/abc123/page12_img0.png"


def test_image_path_to_url_none():
    assert image_path_to_url(None) is None


def test_parse_search_hit_maps_fields():
    hit = {
        "_score": 0.87,
        "_source": {
            "chunk_id": "c1",
            "doc_id": "d1",
            "filename": "huawei.pdf",
            "page_number": 12,
            "chunk_type": "text",
            "content": "Five-Year Financial Highlights revenue",
            "image_path": "data/images/d1/page12_img0.png",
            "extra_metadata": {"extraction_method": "pdfplumber"},
        },
    }
    chunk = parse_search_hit(hit)
    assert chunk.chunk_id == "c1"
    assert chunk.score == 0.87
    assert chunk.image_url == "/images/d1/page12_img0.png"
    assert chunk.extraction_method == "pdfplumber"

import pytest

from contract_rag.ir import BlockType
from contract_rag.parse.paddle_parser import _select_api, lines_to_blocks


def test_lines_to_blocks_sets_bbox_confidence_and_engine():
    lines = [
        {"text": "AGREEMENT", "box": (10, 20, 200, 40), "conf": 0.98, "page": 1},
        {"text": "by Acme Inc.", "box": (10, 50, 200, 70), "conf": 0.91, "page": 1},
    ]
    blocks = lines_to_blocks(lines)
    assert len(blocks) == 2
    assert blocks[0].type is BlockType.PARAGRAPH
    assert blocks[0].source_engine == "paddleocr"
    assert blocks[0].bbox is not None
    assert blocks[0].bbox.page == 1
    assert blocks[1].confidence == 0.91
    assert blocks[0].block_id == "#/ocr/0"


def test_predict_result_to_lines_parses_v3_dict():
    from contract_rag.parse.paddle_parser import predict_result_to_lines

    res = {
        "rec_texts": ["AGREEMENT", "Acme Inc."],
        "rec_scores": [0.98, 0.91],
        "rec_boxes": [(10, 20, 200, 40), (10, 50, 200, 70)],
    }
    lines = predict_result_to_lines(res, page_no=2)
    assert lines == [
        {"text": "AGREEMENT", "box": (10.0, 20.0, 200.0, 40.0), "conf": 0.98, "page": 2},
        {"text": "Acme Inc.", "box": (10.0, 50.0, 200.0, 70.0), "conf": 0.91, "page": 2},
    ]


def test_legacy_result_to_lines_parses_v2_quads():
    from contract_rag.parse.paddle_parser import legacy_result_to_lines

    page = [[[(10, 20), (200, 22), (200, 40), (10, 38)], ("AGREEMENT", 0.97)]]
    lines = legacy_result_to_lines(page, page_no=1)
    assert lines[0]["box"] == (10, 20, 200, 40)
    assert lines[0]["conf"] == 0.97


def test_legacy_result_to_lines_none_page_is_empty():
    from contract_rag.parse.paddle_parser import legacy_result_to_lines

    assert legacy_result_to_lines(None, page_no=1) == []


@pytest.mark.parametrize(
    "version,expected",
    [
        ("3.0.0", "predict"),
        ("3.7.0", "predict"),
        ("4.1.2", "predict"),
        ("2.7.0.3", "ocr"),
        ("2.6.1", "ocr"),
    ],
)
def test_select_api_by_major_version(version, expected):
    assert _select_api(version) == expected


def test_select_api_raises_on_unparseable_version():
    with pytest.raises(ValueError, match="unparseable paddleocr version"):
        _select_api("not-a-version")

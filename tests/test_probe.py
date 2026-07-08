from contract_rag.parse.probe import (
    DocProfile,
    PageProfile,
    _page_char_count,
    profile_from_counts,
    profile_from_pages,
)


class _FakeTextpage:
    def __init__(self, text, raise_on):
        self._text = text
        self._raise_on = raise_on
        self.closed = False

    def get_text_range(self):
        if self._raise_on == "get_text_range":
            raise RuntimeError("decode error")
        return self._text

    def close(self):
        self.closed = True


class _FakePage:
    def __init__(self, text=None, raise_on=None):
        self._text = text
        self._raise_on = raise_on
        self.closed = False

    def get_textpage(self):
        if self._raise_on == "get_textpage":
            raise RuntimeError("corrupt page")
        return _FakeTextpage(self._text, raise_on=self._raise_on)

    def close(self):
        self.closed = True


def test_page_char_count_returns_stripped_text_length():
    page = _FakePage(text="  hello world  ")
    assert _page_char_count(page) == len("hello world")
    assert page.closed


def test_page_char_count_degrades_to_zero_on_textpage_failure():
    # a corrupt page must not crash the whole probe — it reads as scanned instead
    page = _FakePage(raise_on="get_textpage")
    assert _page_char_count(page) == 0
    assert page.closed  # still closed despite the failure


def test_page_char_count_degrades_to_zero_on_text_range_failure():
    page = _FakePage(text="irrelevant", raise_on="get_text_range")
    assert _page_char_count(page) == 0
    assert page.closed


def test_text_coverage_is_fraction_of_pages_with_text():
    p = profile_from_counts(page_count=10, pages_with_text=8)
    assert p == DocProfile(page_count=10, pages_with_text=8, text_coverage=0.8)


def test_zero_pages_is_zero_coverage_not_crash():
    p = profile_from_counts(page_count=0, pages_with_text=0)
    assert p.text_coverage == 0.0


def test_profile_from_pages_matches_doc_level_average():
    # a mixed doc: 2 digital + 2 scanned pages → doc coverage 0.5 (the number that
    # describes no actual page — the whole reason per-page routing exists)
    pages = [
        PageProfile(page=1, char_count=800, has_text=True),
        PageProfile(page=2, char_count=600, has_text=True),
        PageProfile(page=3, char_count=0, has_text=False),
        PageProfile(page=4, char_count=0, has_text=False),
    ]
    prof = profile_from_pages(pages)
    assert prof == DocProfile(page_count=4, pages_with_text=2, text_coverage=0.5)


def test_profile_from_pages_empty():
    assert profile_from_pages([]).text_coverage == 0.0

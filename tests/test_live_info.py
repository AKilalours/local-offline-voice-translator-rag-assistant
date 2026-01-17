from main import _is_live_info_query

def test_live_news():
    assert _is_live_info_query("today's news headlines") is True

def test_non_live():
    assert _is_live_info_query("what is RAG") is False

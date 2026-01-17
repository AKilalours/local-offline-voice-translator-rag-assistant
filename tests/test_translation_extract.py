from main import _extract_translate_request

def test_extract_in_language():
    phrase, lang = _extract_translate_request("how are you in French")
    assert phrase.lower() == "how are you"
    assert lang == "French"

def test_extract_translate_prefix():
    phrase, lang = _extract_translate_request("translate, how are you in French")
    assert phrase.lower() == "how are you"
    assert lang == "French"

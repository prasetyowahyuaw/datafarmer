from datafarmer.utils import parse_text

def test_parse_text():
    text = """
    ```json
    {
        "text": "this is a test"
    }
    ```
    """
    result = parse_text(text)
    assert result == {"text": "this is a test"}
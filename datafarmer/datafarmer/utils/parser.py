import re
import json

def parse_text(text: str, format: str="json"):
    """parses the response text from the generative response

    Args:
        response (str): response string
        format (str, optional): the format of the response. Defaults to "json".
    Returns:
        Dict: parsed text
    """
    
    assert format in ["json"], f"Invalid format. currently doesn't support {format} format"

    if format == "json":
        # capture the json markdown code block
        match = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        json_text = match[-1].strip()
        return json.loads(json_text)
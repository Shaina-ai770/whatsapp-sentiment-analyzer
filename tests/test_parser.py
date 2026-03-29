from src.parser import parse_chat
import io


def test_parse_simple(tmp_path):
    content = """12/31/20, 9:12 PM - Alice: Hello\n12/31/20, 9:13 PM - Bob: Hi Alice\n"""
    p = tmp_path / "chat.txt"
    p.write_text(content)
    df = parse_chat(str(p))
    assert df.shape[0] == 2
    assert df.loc[0, 'author'] == 'Alice'
    assert 'Hello' in df.loc[0, 'message']
    # mobile column should exist and be None for these entries
    assert 'mobile' in df.columns
    assert df.loc[0, 'mobile'] is None

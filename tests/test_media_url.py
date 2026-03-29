from src.parser import parse_chat
from src.preprocess import preprocess_df


def test_media_and_url(tmp_path):
    content = """12/31/20, 9:12 PM - Alice: Check this out https://example.com/page\n12/31/20, 9:13 PM - Bob: IMG-20201010-WA0001.jpg\n"""
    p = tmp_path / "chat.txt"
    p.write_text(content)
    df = parse_chat(str(p))
    df = preprocess_df(df)
    assert 'media' in df.columns
    assert df.loc[1, 'media'] is not None
    assert df.loc[0, 'urls'] and 'https://example.com/page' in df.loc[0, 'urls'][0]

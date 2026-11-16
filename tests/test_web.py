from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"


def test_page_has_a_textarea_inside_a_form():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert "<textarea" in html
    assert "<form" in html


def test_page_carries_no_credentials():
    for name in ("index.html", "app.js"):
        text = (WEB / name).read_text(encoding="utf-8")
        assert "ANTHROPIC" not in text.upper()
        assert "sk-ant-" not in text
        assert "api_key" not in text.lower()


def test_assets_the_page_references_exist():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert "/static/app.js" in html
    assert (WEB / "app.js").is_file()
    assert (WEB / "styles.css").is_file()


def test_send_area_announces_status_to_screen_readers():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert "aria-live" in html


def test_dark_tokens_match_the_portfolio_palette():
    css = (WEB / "styles.css").read_text(encoding="utf-8").replace(" ", "")
    assert "--bg:#0a0f1c" in css
    assert "--primary:#0ea5e9" in css


def test_app_js_renders_file_and_page_and_scrolls_to_a_source():
    js = (WEB / "app.js").read_text(encoding="utf-8")
    assert "source-file" in js
    assert "source-page" in js
    assert "scrollIntoView" in js


def test_app_js_surfaces_the_server_rate_limit_detail_verbatim():
    js = (WEB / "app.js").read_text(encoding="utf-8")
    assert "429" in js
    assert "detail" in js

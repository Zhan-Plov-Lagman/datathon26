"""Brand video playback must always leave the application accessible."""
import os

from playwright.sync_api import expect, sync_playwright

from app.api.server import create_public_app
from tests.api.conftest import settings  # noqa: F401
from tests.browser.test_ui import server


def test_intro_playback_reload_skip_replay_and_reduced_motion(settings):
    with server(create_public_app(settings)) as url, sync_playwright() as p:
        browser = p.chromium.launch(channel=os.environ.get("FORTE_BROWSER_CHANNEL", "msedge"))
        try:
            page = browser.new_page(viewport={"width": 390, "height": 844})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url)
            expect(page.locator("#brand-intro")).to_be_visible()
            page.wait_for_function("() => document.querySelector('.intro-video')?.currentTime > 0.1")
            assert page.locator(".intro-video").evaluate("v => v.muted && v.playsInline")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            # Real media end, not a synthetic event, must release the dialog and decoder.
            expect(page.locator("#brand-intro")).not_to_be_visible(timeout=12000)
            expect(page.locator(".intro-video")).to_have_count(0)
            page.reload()
            expect(page.locator("#brand-intro")).to_be_visible()
            page.locator("#skip-intro").click()
            expect(page.locator("#brand-intro")).not_to_be_visible()
            page.goto(url + "#workspace")
            page.reload()  # A hash-only navigation does not reload the document.
            expect(page.locator("#brand-intro")).to_be_visible()
            page.locator("#skip-intro").click()
            expect(page.locator("#brand-intro")).not_to_be_visible()
            page.locator("#replay-intro").click()
            expect(page.locator("#brand-intro")).to_be_visible()
            page.keyboard.press("Escape")
            expect(page.locator("#brand-intro")).not_to_be_visible()
            expect(page.locator("#replay-intro")).to_be_focused()
            page.locator("#replay-intro").click()
            page.locator("#skip-intro").click()
            expect(page.locator("#brand-intro")).not_to_be_visible()
            assert page.evaluate("!document.documentElement.classList.contains('intro-open')")
            reduced = browser.new_page(reduced_motion="reduce")
            media_requests = []
            reduced.on("request", lambda request: media_requests.append(request.url)
                       if "forte-intro.mp4" in request.url else None)
            reduced.goto(url)
            expect(reduced.locator("#brand-intro")).not_to_be_visible()
            assert media_requests == []
            assert errors == []
        finally:
            browser.close()


def test_intro_releases_site_on_missing_media_and_autoplay_rejection(settings):
    with server(create_public_app(settings)) as url, sync_playwright() as p:
        browser = p.chromium.launch(channel=os.environ.get("FORTE_BROWSER_CHANNEL", "msedge"))
        try:
            for failure in ("missing", "autoplay"):
                page = browser.new_page()
                if failure == "missing":
                    page.route("**/media/forte-intro.mp4*", lambda route: route.abort())
                else:
                    page.add_init_script("HTMLMediaElement.prototype.play = () => Promise.reject(new Error('blocked'))")
                page.goto(url)
                expect(page.locator("#brand-intro")).not_to_be_visible(timeout=5000)
                expect(page.locator(".intro-video")).to_have_count(0)
                page.locator("#api-key").fill("synthetic-test-key")
                assert page.evaluate("!document.documentElement.classList.contains('intro-open')")
                page.close()
        finally:
            browser.close()

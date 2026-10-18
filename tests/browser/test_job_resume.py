"""Elapsed time and reload recovery through real HTTP, using synthetic MOV only."""
import asyncio
import os
import re

from playwright.sync_api import expect, sync_playwright

from app.api.schema import AnalysisResult
from app.api.server import create_public_app
from app.jobs.store import JobStore
from tests.api.conftest import result, settings  # noqa: F401
from tests.browser.test_ui import server, synthetic_mov  # noqa: F401


def test_elapsed_upload_reload_reauthentication_and_frozen_result(settings, result, synthetic_mov):
    store = JobStore(settings)
    store.hello('timer-test', True, 'synthetic-only')
    app = create_public_app(settings, store)
    uploads = []
    job_reads = []

    @app.middleware('http')
    async def slow_upload(request, call_next):
        if request.url.path.startswith('/api/v1/jobs/'):
            job_reads.append(request.url.path)
        if request.method == 'POST' and request.url.path == '/api/v1/analyze':
            uploads.append(True)
            await asyncio.sleep(2.2)
        return await call_next(request)

    with server(app) as url, sync_playwright() as p:
        browser = p.chromium.launch(channel=os.environ.get('FORTE_BROWSER_CHANNEL', 'msedge'))
        try:
            page = browser.new_page(reduced_motion='reduce')
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(url)
            page.locator('#api-key').fill(settings.api_key)
            page.locator('#video-file').set_input_files(str(synthetic_mov))
            page.wait_for_function('() => Number.isFinite(document.querySelector("#player").duration)')
            page.locator('#submit').click()
            expect(page.locator('#status-text')).to_contain_text('Загружаем видео')
            expect(page.locator('#elapsed-status')).to_be_visible()
            page.evaluate('''() => {
                document.querySelector('#job-id').value = 'other-synthetic';
                document.querySelector('#resume-form').requestSubmit();
            }''')
            expect(page.locator('#status-text')).to_contain_text('Загружаем видео')
            page.wait_for_function('() => Number(document.querySelector("#elapsed-value").textContent) >= 2')
            expect(page.locator('#job-label')).to_contain_text('ID:')
            expect(page.locator('#submit')).to_be_disabled()
            job_id = page.locator('#job-id').input_value()
            expect(page).to_have_url(url + '/#job=' + job_id)
            saved = page.evaluate('JSON.parse(sessionStorage.getItem("forte-job-session"))')
            assert saved['filename'] == 'synthetic.mov'
            assert round(saved['durationSec']) == 3
            assert saved['startedWallMs'] > 0
            page.reload()
            expect(page.locator('#api-key')).to_have_value('')
            expect(page.locator('#status-text')).to_contain_text('Введите ключ доступа')
            expect(page.locator('#job-id')).to_have_value(job_id)
            page.locator('#api-key').fill('wrong')
            page.locator('#check-ready').click()
            expect(page.locator('#readiness')).to_contain_text('Ключ доступа не принят')
            assert page.evaluate('JSON.parse(sessionStorage.getItem("forte-job-session")).jobId') == job_id
            page.locator('#api-key').fill(settings.api_key)
            page.locator('#api-key').press('Tab')
            expect(page.locator('#status-text')).to_contain_text('Ожидаем результат сервера')
            lease = store.claim('timer-test')
            assert lease['job_id'] == job_id
            store.complete(job_id, 'timer-test', lease['attempt_token'], AnalysisResult.model_validate(result))
            expect(page.locator('#status-badge')).to_have_text('Завершено', timeout=10000)
            expect(page.locator('#submit')).to_be_enabled()
            expect(page.locator('#result-meta')).to_contain_text('synthetic.mov · видео 3 с · до результата')
            expect(page.locator('#result-meta')).to_contain_text(re.compile(r'\d{2}\.\d{2}, \d{2}:\d{2}'))
            frozen = page.locator('#elapsed-value').inner_text()
            page.wait_for_timeout(1200)
            expect(page.locator('#elapsed-value')).to_have_text(frozen)
            storage = page.evaluate('JSON.stringify({local: {...localStorage}, session: {...sessionStorage}})')
            assert settings.api_key not in storage
            assert 'image_base64' not in storage and 'violations' not in storage
            page.reload()
            expect(page.locator('#elapsed-value')).to_have_text(frozen)
            page.wait_for_timeout(1200)
            expect(page.locator('#elapsed-value')).to_have_text(frozen)
            page.locator('#api-key').fill(settings.api_key)
            page.locator('#check-ready').click()
            expect(page.locator('#status-badge')).to_have_text('Завершено')
            expect(page.locator('.event')).to_have_count(1)
            assert uploads == [True]
            assert '/api/v1/jobs/other-synthetic' not in job_reads
            assert errors == []
        finally:
            browser.close()


def test_job_url_without_timing_and_repeated_key_checks_do_not_duplicate_polling(settings, result, synthetic_mov):
    store = JobStore(settings)
    store.create('shared-synthetic')
    app = create_public_app(settings, store)
    polls = []

    @app.middleware('http')
    async def count_polls(request, call_next):
        if request.url.path == '/api/v1/jobs/shared-synthetic':
            polls.append(True)
            await asyncio.sleep(.2)
        return await call_next(request)

    with server(app) as url, sync_playwright() as p:
        browser = p.chromium.launch(channel=os.environ.get('FORTE_BROWSER_CHANNEL', 'msedge'))
        try:
            page = browser.new_page(reduced_motion='reduce')
            page.goto(url + '/#job=shared-synthetic')
            expect(page.locator('#job-id')).to_have_value('shared-synthetic')
            expect(page.locator('#elapsed-status')).to_be_hidden()
            page.locator('#api-key').fill(settings.api_key)
            page.locator('#check-ready').click()
            page.locator('#check-ready').click()
            expect(page.locator('#status-text')).to_contain_text('Ожидаем результат сервера')
            page.wait_for_timeout(400)
            assert len(polls) == 1
            store.hello('shared-test', True, 'synthetic-only')
            lease = store.claim('shared-test')
            store.complete('shared-synthetic', 'shared-test', lease['attempt_token'], AnalysisResult.model_validate(result))
            expect(page.locator('#status-badge')).to_have_text('Завершено', timeout=10000)
            expect(page.locator('#result-meta')).to_contain_text('время до результата неизвестно')
            expect(page.locator('#elapsed-status')).to_be_hidden()
            assert page.locator('#api-key').input_value() not in page.url
            page.evaluate('''() => {
                document.querySelector('#job-id').value = 'missing-synthetic';
                document.querySelector('#resume-form').requestSubmit();
            }''')
            expect(page.locator('#status-text')).to_contain_text('Задание с таким ID не найдено')
            page.locator('#video-file').set_input_files(str(synthetic_mov))
            expect(page.locator('#submit')).to_be_enabled()
            expect(page.locator('#job-id')).to_have_value('')
        finally:
            browser.close()

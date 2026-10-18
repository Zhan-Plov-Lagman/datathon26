"""Synthetic inference responses; real MOV, v2 adapter, SQLite queue and HTTP transport."""
import json
from pathlib import Path
import time
from unittest.mock import Mock

import pytest

from app.jobs.unified import UnifiedProcessor
from app.jobs.worker import execute_job
from scripts import run_local
from tests.api.test_api import clients,claim,upload
from tests.worker.test_unified import analyzer,reviewed_registry,release,movie


class InlineProcess:
    """Replace process isolation only; run the actual processor and rules."""
    def __init__(self,processor):
        self.processor=processor
    def run(self,task,deadline,lost):
        try:
            result=self.processor.process(task['video'],task['job_id'],task['timeout_sec'])
            Path(task['output']).write_text(json.dumps(result),encoding='utf-8')
            return True
        except Exception:
            return False


@pytest.mark.parametrize('mode,expected',[
    ('positive','violation'),('negative','no_violation'),('failed',None)])
def test_real_http_queue_processor_and_jpeg(settings,tmp_path,movie,mode,expected):
    model=analyzer(mode)
    registry=reviewed_registry()
    processor=InlineProcess(UnifiedProcessor(model,registry,release(model,registry)))
    public,private=clients(settings)
    with public,private:
        response=upload(public,content=movie.read_bytes())
        assert response.status_code==202
        job_id=response.json()['job_id']
        lease=claim(private)
        assert execute_job(private,processor,lease,'gpu-a',tmp_path,time.monotonic())==(expected is not None)
        body=public.get(f'/api/v1/jobs/{job_id}').json()
        assert body['status']==('failed' if expected is None else 'done')
        if expected is None:
            assert 'verdict' not in body and 'violations' not in body
        else:
            assert body['verdict']==expected
            assert bool(body['violations'])==(expected=='violation')
            if body['violations']:
                assert body['violations'][0]['evidence_frames'][0]['timestamp_sec']==2.9


def test_unified_supervisor_requires_release(monkeypatch):
    monkeypatch.setattr(run_local,'load_dotenv',lambda **kwargs:None)
    monkeypatch.setattr(run_local.Settings,'from_env',lambda:None)
    monkeypatch.setenv('FORTE_PROCESSOR_FACTORY','app.jobs.unified:create_processor')
    monkeypatch.delenv('FORTE_UNIFIED_RELEASE',raising=False)
    spawn=Mock()
    monkeypatch.setattr(run_local.subprocess,'Popen',spawn)
    assert run_local.main()==2
    spawn.assert_not_called()


def test_custom_processor_does_not_need_old_evaluator_factory(monkeypatch):
    monkeypatch.setattr(run_local,'load_dotenv',lambda **kwargs:None)
    monkeypatch.setattr(run_local.Settings,'from_env',lambda:None)
    monkeypatch.setenv('FORTE_PROCESSOR_FACTORY','app.jobs.unified:create_processor')
    monkeypatch.setenv('FORTE_UNIFIED_RELEASE','synthetic-test-only.json')
    monkeypatch.delenv('FORTE_RULES_FACTORY',raising=False)
    monkeypatch.setattr(run_local,'require_free_ports',lambda:None)
    monkeypatch.setattr(run_local,'wait_for_coordinator',lambda process:None)
    coordinator,worker=Mock(),Mock()
    coordinator.poll.return_value,worker.poll.return_value=None,1
    spawn=Mock(side_effect=[coordinator,worker])
    monkeypatch.setattr(run_local.subprocess,'Popen',spawn)
    monkeypatch.setattr(run_local,'stop_process',Mock())
    assert run_local.main()==1
    assert spawn.call_count==2

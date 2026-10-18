"""Run the existing synthetic smoke with outbound Python sockets blocked.

This is a diagnostic wrapper, not another inference implementation. Use a
different output directory for every model/run so reports are not overwritten.
It deliberately accepts no video input: the smoke generates its own fixture.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import socket
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    model = args.model_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_PROGRESS_BARS='1')

    def deny(*_args, **_kwargs):
        raise RuntimeError('Outbound socket blocked during synthetic offline smoke')

    socket.create_connection = deny
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    sys.path.insert(0, str(root))
    os.chdir(output)
    sys.argv = [str(root / 'scripts/smoke_worker.py'), '--model-dir', str(model)]
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()

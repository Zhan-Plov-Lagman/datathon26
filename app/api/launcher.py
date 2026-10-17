"""Importable multiprocessing target (package __main__ targets fail under Windows spawn)."""
import uvicorn


def serve(factory, host, port):
    uvicorn.run(factory, factory=True, host=host, port=port, access_log=False,
                log_level="warning", proxy_headers=False, server_header=False)

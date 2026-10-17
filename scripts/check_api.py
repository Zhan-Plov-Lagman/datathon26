"""Read-only external checks. Prints status codes only, never keys or scene content."""
import argparse
import os

import httpx
from dotenv import load_dotenv


def main():
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="Team-controlled HTTPS URL (or localhost for local test)")
    args = parser.parse_args()
    checks = [("health", "GET", "/healthz", {}, 200),
              ("anonymous_get", "GET", "/api/v1/jobs/check-only", {}, 401),
              ("anonymous_post", "POST", "/api/v1/analyze", {}, 401),
              ("wrong_key", "GET", "/api/v1/jobs/check-only", {"X-API-Key": "wrong"}, 401),
              ("private_routes_hidden", "POST", "/internal/jobs/claim", {}, 404)]
    key = os.environ.get("FORTE_API_KEY")
    if key:
        checks.append(("authorized_unknown_job", "GET", "/api/v1/jobs/check-only", {"X-API-Key": key}, 404))
    failed = 0
    with httpx.Client(base_url=args.base_url, timeout=15, trust_env=False, follow_redirects=False) as client:
        for name, method, path, headers, expected in checks:
            try:
                status = client.request(method, path, headers=headers).status_code
            except httpx.HTTPError:
                status = "connection_error"
            passed = status == expected
            failed += not passed
            print(f"{name}: {status}; {'PASS' if passed else 'FAIL'}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

"""Generate local keys once without displaying them or overwriting an existing .env."""
import secrets
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / ".env"
    if target.exists():
        print("Existing .env preserved.")
        return
    template = (root / ".env.example").read_text(encoding="utf-8")
    template = template.replace("FORTE_API_KEY=\n", f"FORTE_API_KEY={secrets.token_urlsafe(32)}\n")
    template = template.replace("FORTE_WORKER_KEY=\n", f"FORTE_WORKER_KEY={secrets.token_urlsafe(32)}\n")
    with target.open("x", encoding="utf-8") as file:
        file.write(template)
    print("Created .env with distinct keys. Read them locally; never commit this file.")


if __name__ == "__main__":
    main()

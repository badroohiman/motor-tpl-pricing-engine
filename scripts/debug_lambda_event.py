from __future__ import annotations

import json
from pathlib import Path

from aws_lambda.handler import lambda_handler


def main() -> None:
    event_path = Path("configs/sample_event.json")
    event = json.loads(event_path.read_text(encoding="utf-8"))
    result = lambda_handler(event, None)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


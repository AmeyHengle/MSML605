"""Entry point: uv run python -m ml605_slack

Per D-01: Socket Mode with SLACK_APP_TOKEN.
"""
from __future__ import annotations

import os
import sys

from slack_bolt.adapter.socket_mode import SocketModeHandler

from ml605_slack.bot import create_app


def main() -> None:
    """Start the Slack bot via SocketModeHandler."""
    bot_token = os.environ.get("SLACK_BOT_TOKEN")
    app_token = os.environ.get("SLACK_APP_TOKEN")

    if not bot_token or not app_token:
        print("ERROR: SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set.", file=sys.stderr)
        sys.exit(1)

    app = create_app()
    handler = SocketModeHandler(app, app_token)
    print("[ml605_slack] Bot starting via Socket Mode...")
    handler.start()


if __name__ == "__main__":
    main()

import os
import requests

def send_slack_alert(message: str, level: str = "info"):
    webhook_url = os.getenv("SLACK_API_KEY")
    if not webhook_url:
        raise RuntimeError("SLACK_WEBHOOK_URL not set in environment")

    colors = {
        "info": "#36a64f",
        "warning": "#ffae42",
        "critical": "#ff0000",
    }
    color = colors.get(level, "#36a64f")

    payload = {
        "attachments": [
            {
                "color": color,
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*Alert:* {message}"}
                    },
                    {
                        "type": "context",
                        "elements": [
                            {"type": "mrkdwn", "text": "Built with *Student Chakra*"}
                        ]
                    }
                ]
            }
        ]
    }

    resp = requests.post(webhook_url, json=payload)

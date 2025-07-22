import subprocess
import requests

def get_git_branch():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ).decode("utf-8").strip()
    except:
        return "unknown-branch"

def send_discord(message, webhook_url, username="EchoBot", emoji="🔔"):
    """
    傳送文字訊息到 Discord Webhook。
    
    Args:
        message (str): 要發送的訊息內容
        webhook_url (str): Discord Webhook URL
        username (str): 顯示的機器人名稱（可選）
        emoji (str): 訊息前綴 emoji（可選）
    """
    if webhook_url is None:
        print("[Warning] Webhook URL is None, skipping Discord notification.")
        return

    payload = {
        "username": username,
        "content": f"{emoji} {message}",
    }

    try:
        response = requests.post(webhook_url, json=payload)
        if response.status_code != 204:
            print(f"[Discord Error] Failed to send: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[Discord Exception] {e}")

import requests
import os
import time
import json

REPORT_PATH = 'reports/quality/quality_report.md'
LOG_PATH = 'logs/automation/sync_quality_report.log'

platforms = [
    {"name": "WIKI", "url": os.environ.get(
        "WIKI_API_URL"), "token": os.environ.get("WIKI_API_TOKEN")},
    {"name": "NOTION", "url": os.environ.get(
        "NOTION_API_URL"), "token": os.environ.get("NOTION_API_TOKEN")},
    {"name": "YUQUE", "url": os.environ.get(
        "YUQUE_API_URL"), "token": os.environ.get("YUQUE_API_TOKEN")},
]

ALERT_WEBHOOK = os.environ.get("ALERT_WEBHOOK")
MAX_RETRIES = 3
RETRY_INTERVAL = 5


def log_event(event_type, detail, level='INFO'):
    event = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'event_type': event_type,
        'level': level,
        'detail': detail
    }
    print(json.dumps(event, ensure_ascii=False))
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(event, ensure_ascii=False) + '\n')


def send_alert(msg):
    log_event('alert', {'msg': msg}, level='ERROR')
    if ALERT_WEBHOOK:
        try:
            requests.post(ALERT_WEBHOOK, json={"text": msg})
        except Exception as e:
            log_event('alert_error', {'msg': str(e)}, level='ERROR')


if not os.path.exists(REPORT_PATH):
    msg = f'[ERROR] 质量报告文件不存在: {REPORT_PATH}'
    log_event('report_missing', {'msg': msg}, level='ERROR')
    send_alert(msg)
    exit(1)

with open(REPORT_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

for p in platforms:
    if p["url"] and p["token"]:
        for attempt in range(1, MAX_RETRIES+1):
            log_event('sync_start', {'platform': p['name'], 'attempt': attempt})
            try:
                resp = requests.post(
                    p["url"],
                    json={"content": content},
                    headers={"Authorization": f"Bearer {p['token']}"}
                )
                if resp.status_code == 200:
                    log_event('sync_success', {'platform': p['name'], 'attempt': attempt})
                    print(f'[OK] 质量报告已成功同步到{p["name"]}')
                    break
                else:
                    msg = f'同步到{p["name"]}失败，状态码: {resp.status_code}, 响应: {resp.text}'
                    log_event('sync_fail', {'platform': p['name'],
                              'attempt': attempt, 'msg': msg}, level='ERROR')
                    if attempt == MAX_RETRIES:
                        send_alert(f'质量报告同步到{p["name"]}失败，状态码: {resp.status_code}')
            except Exception as e:
                log_event('sync_exception', {
                          'platform': p['name'], 'attempt': attempt, 'error': str(e)}, level='ERROR')
                if attempt == MAX_RETRIES:
                    send_alert(f'质量报告同步到{p["name"]}异常: {e}')
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_INTERVAL)
    else:
        log_event('sync_skip', {'platform': p['name']}, level='WARNING')
        print(f'[SKIP] 未配置{p["name"]}平台API，跳过同步')

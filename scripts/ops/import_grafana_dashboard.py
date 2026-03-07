import requests
import os

GRAFANA_URL = os.environ.get('GRAFANA_URL', 'http://localhost:3000')
GRAFANA_API_KEY = os.environ.get('GRAFANA_API_KEY', 'your_grafana_api_key')
DASHBOARD_JSON_PATH = 'dashboards/automation_dashboard.json'

if not os.path.exists(DASHBOARD_JSON_PATH):
    print(f'[ERROR] 仪表盘模板文件不存在: {DASHBOARD_JSON_PATH}')
    exit(1)

with open(DASHBOARD_JSON_PATH, 'r', encoding='utf-8') as f:
    dashboard = f.read()

resp = requests.post(
    f'{GRAFANA_URL}/api/dashboards/db',
    headers={'Authorization': f'Bearer {GRAFANA_API_KEY}', 'Content-Type': 'application/json'},
    data=dashboard
)
print('导入结果:', resp.status_code, resp.text)

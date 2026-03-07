#!/usr/bin/env python3

with open('src/infrastructure/resource/resource_dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace system data append
system_old = '        self.data["system"].append({})'
system_new = '''        self.data["system"].append({
            "timestamp": current_time,
            "cpu": {"percent": random.uniform(20, 80)},
            "memory": {"percent": random.uniform(30, 90)},
            "disk": {"percent": random.uniform(10, 70)}
        })'''

content = content.replace(system_old, system_new)

# Replace GPU data append
gpu_old = '        self.data["gpu"].append({})'
gpu_new = '''        self.data["gpu"].append({
            "timestamp": current_time,
            "gpus": [
                {
                    "index": 0,
                    "name": "Mock GPU",
                    "memory": {"allocated": random.uniform(1000, 8000)},
                    "utilization": random.uniform(10, 90),
                    "temperature": random.uniform(40, 80)
                }
            ]
        })'''

content = content.replace(gpu_old, gpu_new)

with open('src/infrastructure/resource/resource_dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Dictionary syntax fixed!")

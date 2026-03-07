# 生产环境安全运维手册

## 概述

本文档详细说明RQA2025量化交易系统在生产环境中的安全运维要求、操作流程和安全管控措施，确保系统在生产环境中的安全运行。

## 安全运维原则

### 1. 最小权限原则

#### 1.1 用户权限管理

```bash
# 创建专门的用户账户
sudo useradd -m -s /bin/bash rqa2025
sudo usermod -aG sudo rqa2025  # 只在必要时

# 设置文件权限
sudo chown -R rqa2025:rqa2025 /opt/rqa2025
sudo chmod -R 750 /opt/rqa2025

# 敏感文件权限
sudo chmod 600 /etc/rqa2025/config.yaml
sudo chmod 600 /etc/rqa2025/database.yaml
sudo chmod 400 /etc/rqa2025/ssl/*.key
```

#### 1.2 服务权限隔离

```bash
# 为不同服务创建独立用户
sudo useradd -r -s /bin/false rqa2025-app
sudo useradd -r -s /bin/false rqa2025-db
sudo useradd -r -s /bin/false rqa2025-cache

# systemd服务权限配置
cat > /etc/systemd/system/rqa2025.service << EOF
[Service]
User=rqa2025-app
Group=rqa2025-app
NoNewPrivileges=true
PrivateTmp=true
PrivateDevices=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/rqa2025/data /var/log/rqa2025
EOF
```

### 2. 访问控制

#### 2.1 网络访问控制

```bash
# 防火墙配置
sudo ufw default deny incoming
sudo ufw default allow outgoing

# 允许特定端口
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 8000:8002/tcp  # 应用端口

# 限制SSH访问
sudo ufw allow from 192.168.1.0/24 to any port 22

# 启用防火墙
sudo ufw --force enable
```

#### 2.2 SSH安全加固

```bash
# /etc/ssh/sshd_config 安全配置
cat >> /etc/ssh/sshd_config << EOF
# 安全配置
Port 22
Protocol 2
HostKey /etc/ssh/ssh_host_rsa_key
HostKey /etc/ssh/ssh_host_ecdsa_key
HostKey /etc/ssh/ssh_host_ed25519_key

# 认证限制
MaxAuthTries 3
MaxStartups 3:30:10
LoginGraceTime 30

# 密码认证
PasswordAuthentication no
PermitEmptyPasswords no
PermitRootLogin no

# 密钥认证
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys

# 会话控制
ClientAliveInterval 300
ClientAliveCountMax 0
TCPKeepAlive no

# 其他安全设置
X11Forwarding no
AllowTcpForwarding no
PermitTTY no
EOF

sudo systemctl reload sshd
```

#### 2.3 堡垒机配置

```bash
# 安装和配置堡垒机
sudo apt-get install teleport

# 堡垒机配置
cat > /etc/teleport.yaml << EOF
teleport:
  data_dir: /var/lib/teleport
  log:
    output: /var/log/teleport.log
    severity: INFO

auth_service:
  enabled: true
  cluster_name: rqa2025-cluster
  tokens:
  - "proxy,node,app:your-token"

proxy_service:
  enabled: true
  listen_addr: 0.0.0.0:3080
  web_listen_addr: 0.0.0.0:3080
  public_addr: teleport.rqa2025.com:3080

ssh_service:
  enabled: true
EOF

sudo systemctl restart teleport
```

### 3. 数据安全

#### 3.1 敏感数据保护

```python
# 数据脱敏函数
import hashlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class DataProtection:
    def __init__(self, master_key):
        self.master_key = master_key
        self.salt = b'rqa2025_salt_2024'

    def encrypt_sensitive_data(self, data: str) -> str:
        """加密敏感数据"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        f = Fernet(key)
        encrypted = f.encrypt(data.encode())
        return encrypted.decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """解密敏感数据"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        f = Fernet(key)
        decrypted = f.decrypt(encrypted_data.encode())
        return decrypted.decode()

    def hash_data(self, data: str) -> str:
        """哈希数据（不可逆）"""
        return hashlib.sha256(data.encode()).hexdigest()

    def mask_pii(self, data: str, data_type: str) -> str:
        """脱敏个人身份信息"""
        if data_type == 'phone':
            return data[:3] + '****' + data[-4:]
        elif data_type == 'email':
            parts = data.split('@')
            return parts[0][0] + '***' + '@' + parts[1]
        elif data_type == 'id_card':
            return data[:6] + '********' + data[-2:]
        else:
            return data
```

#### 3.2 数据库安全

```sql
-- PostgreSQL安全配置
-- 启用SSL连接
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_cert_file = '/etc/ssl/certs/rqa2025.crt';
ALTER SYSTEM SET ssl_key_file = '/etc/ssl/private/rqa2025.key';

-- 密码策略
ALTER SYSTEM SET password_encryption = 'scram-sha-256';
ALTER SYSTEM SET row_security = on;

-- 审计配置
CREATE EXTENSION IF NOT EXISTS pgaudit;
ALTER SYSTEM SET pgaudit.log = 'read,write,ddl';
ALTER SYSTEM SET pgaudit.log_relation = on;
ALTER SYSTEM SET pgaudit.log_statement_once = on;

-- 重新加载配置
SELECT pg_reload_conf();
```

#### 3.3 备份安全

```bash
#!/bin/bash
# secure_backup.sh

# 备份加密配置
BACKUP_PASSWORD="your-secure-backup-password"
ENCRYPTION_KEY="/etc/rqa2025/backup.key"

# 生成加密密钥
openssl rand -base64 32 > $ENCRYPTION_KEY
chmod 600 $ENCRYPTION_KEY

# 加密备份
encrypt_backup() {
    local backup_file=$1
    local encrypted_file="${backup_file}.enc"

    openssl enc -aes-256-cbc -salt -in $backup_file -out $encrypted_file -pass file:$ENCRYPTION_KEY

    if [ $? -eq 0 ]; then
        rm $backup_file  # 删除原文件
        echo "备份已加密: $encrypted_file"
    else
        echo "备份加密失败"
        exit 1
    fi
}

# 解密备份
decrypt_backup() {
    local encrypted_file=$1
    local backup_file="${encrypted_file%.enc}"

    openssl enc -d -aes-256-cbc -in $encrypted_file -out $backup_file -pass file:$ENCRYPTION_KEY

    if [ $? -eq 0 ]; then
        echo "备份已解密: $backup_file"
    else
        echo "备份解密失败"
        exit 1
    fi
}

# 使用示例
# 加密备份
encrypt_backup "/data/backup/database/rqa2025_backup.sql.gz"

# 解密备份
decrypt_backup "/data/backup/database/rqa2025_backup.sql.gz.enc"
```

## 安全监控和告警

### 1. 安全事件监控

#### 1.1 入侵检测

```bash
# 安装和配置OSSEC
sudo apt-get install ossec-hids

# OSSEC配置
cat > /var/ossec/etc/ossec.conf << EOF
<ossec_config>
  <global>
    <email_notification>yes</email_notification>
    <email_to>security@rqa2025.com</email_to>
  </global>

  <rules>
    <include>rules_config.xml</include>
  </rules>

  <syscheck>
    <frequency>7200</frequency>
    <directories check_all="yes">/etc,/usr/bin,/usr/sbin</directories>
    <directories check_all="yes">/bin,/sbin</directories>
  </syscheck>

  <log_analysis>
    <localfile>
      <location>/var/log/auth.log</location>
      <log_format>syslog</log_format>
    </localfile>
    <localfile>
      <location>/var/log/rqa2025/security.log</location>
      <log_format>syslog</log_format>
    </localfile>
  </log_analysis>
</ossec_config>
EOF

sudo systemctl restart ossec
```

#### 1.2 安全日志监控

```python
# 安全日志监控器
import re
import time
from datetime import datetime
import logging

class SecurityLogMonitor:
    def __init__(self, log_file):
        self.log_file = log_file
        self.security_patterns = {
            'failed_login': r'Failed login attempt',
            'suspicious_ip': r'(\d+\.){3}\d+',
            'sql_injection': r'(union|select|insert|drop|update|delete).*--',
            'xss_attempt': r'<script.*>.*</script>',
            'file_upload': r'upload.*\.(php|jsp|exe|bat)'
        }
        self.alert_thresholds = {
            'failed_login': 5,  # 5分钟内5次失败登录
            'suspicious_activity': 10  # 5分钟内10次可疑活动
        }

    def monitor_logs(self):
        """监控安全日志"""
        try:
            with open(self.log_file, 'r') as f:
                f.seek(0, 2)  # 跳转到文件末尾
                while True:
                    line = f.readline()
                    if line:
                        self.analyze_log_line(line.strip())
                    time.sleep(0.1)
        except Exception as e:
            logging.error(f"日志监控错误: {str(e)}")

    def analyze_log_line(self, line):
        """分析日志行"""
        timestamp = datetime.now().isoformat()

        # 检查各种安全模式
        for pattern_name, pattern in self.security_patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                self.handle_security_event(pattern_name, line, timestamp)

    def handle_security_event(self, event_type, line, timestamp):
        """处理安全事件"""
        logging.warning(f"安全事件检测: {event_type} - {line}")

        # 根据事件类型采取相应措施
        if event_type == 'failed_login':
            self.handle_failed_login(line)
        elif event_type == 'sql_injection':
            self.handle_sql_injection(line)
        elif event_type == 'xss_attempt':
            self.handle_xss_attempt(line)

        # 发送告警
        self.send_security_alert(event_type, line, timestamp)

    def handle_failed_login(self, line):
        """处理登录失败"""
        # 提取IP地址
        ip_match = re.search(r'(\d+\.){3}\d+', line)
        if ip_match:
            ip = ip_match.group()
            # 检查是否应该封锁IP
            if self.should_block_ip(ip):
                self.block_ip(ip)

    def handle_sql_injection(self, line):
        """处理SQL注入尝试"""
        # 记录攻击源
        ip_match = re.search(r'(\d+\.){3}\d+', line)
        if ip_match:
            ip = ip_match.group()
            logging.critical(f"SQL注入尝试来自: {ip}")

    def handle_xss_attempt(self, line):
        """处理XSS尝试"""
        # 记录攻击源
        ip_match = re.search(r'(\d+\.){3}\d+', line)
        if ip_match:
            ip = ip_match.group()
            logging.critical(f"XSS尝试来自: {ip}")

    def should_block_ip(self, ip):
        """判断是否应该封锁IP"""
        # 实现IP封锁逻辑
        return False  # 暂时不封锁

    def block_ip(self, ip):
        """封锁IP地址"""
        # 使用iptables封锁IP
        import subprocess
        try:
            subprocess.run(['sudo', 'iptables', '-A', 'INPUT', '-s', ip, '-j', 'DROP'])
            logging.info(f"已封锁IP: {ip}")
        except Exception as e:
            logging.error(f"封锁IP失败: {ip} - {str(e)}")

    def send_security_alert(self, event_type, line, timestamp):
        """发送安全告警"""
        # 这里可以集成邮件、钉钉、微信等告警方式
        alert_message = f"""
安全告警通知
时间: {timestamp}
类型: {event_type}
详情: {line}
"""

        logging.critical(alert_message)

        # 发送邮件告警（示例）
        # self.send_email_alert(event_type, line, timestamp)
```

### 2. 安全审计

#### 2.1 系统安全审计

```bash
#!/bin/bash
# security_audit.sh

echo "=== RQA2025 安全审计报告 ==="
echo "审计时间: $(date)"

# 1. 用户和权限审计
echo "1. 用户和权限审计"
echo "系统用户列表:"
awk -F: '{print $1}' /etc/passwd

echo "Sudo用户列表:"
grep -Po '^sudo.+:\K.*$' /etc/group

echo "空密码用户检查:"
awk -F: '($2 == "") {print $1}' /etc/shadow

# 2. 文件权限审计
echo "2. 文件权限审计"
echo "敏感文件权限检查:"
ls -la /etc/passwd /etc/shadow /etc/sudoers
ls -la /etc/rqa2025/

# 3. 网络安全审计
echo "3. 网络安全审计"
echo "开放端口检查:"
netstat -tlnp | grep LISTEN

echo "防火墙状态:"
sudo ufw status

# 4. 服务安全审计
echo "4. 服务安全审计"
echo "运行中的服务:"
systemctl list-units --type=service --state=running

echo "未授权的服务:"
# 检查是否有未授权的网络服务

# 5. 日志审计
echo "5. 日志审计"
echo "最近的登录记录:"
last -10

echo "失败的登录尝试:"
grep "Failed password" /var/log/auth.log | tail -10

# 6. 安全配置审计
echo "6. 安全配置审计"
echo "SSH配置检查:"
grep -E "^(Port|PermitRootLogin|PasswordAuthentication)" /etc/ssh/sshd_config

echo "密码策略检查:"
grep -E "^(PASS_MAX_DAYS|PASS_MIN_DAYS|PASS_WARN_AGE)" /etc/login.defs

echo "=== 安全审计完成 ==="
```

#### 2.2 应用安全审计

```python
# application_security_audit.py

import os
import re
import json
from pathlib import Path
from datetime import datetime

class ApplicationSecurityAudit:
    def __init__(self, app_directory):
        self.app_directory = Path(app_directory)
        self.findings = []

    def audit_code_security(self):
        """代码安全审计"""
        python_files = list(self.app_directory.rglob('*.py'))

        security_issues = {
            'hardcoded_password': r'password\s*=\s*[\'"][^\'"]+[\'"]',
            'hardcoded_secret': r'secret\s*=\s*[\'"][^\'"]+[\'"]',
            'hardcoded_api_key': r'api_key\s*=\s*[\'"][^\'"]+[\'"]',
            'sql_injection_risk': r'execute\(.*\+.*\)',
            'debug_enabled': r'debug\s*=\s*True',
            'insecure_random': r'random\.',
            'weak_crypto': r'md5\(|sha1\('
        }

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                for issue_type, pattern in security_issues.items():
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        self.findings.append({
                            'type': 'code_security',
                            'issue': issue_type,
                            'file': str(file_path),
                            'line': content[:match.start()].count('\n') + 1,
                            'code': match.group().strip(),
                            'severity': 'HIGH' if issue_type in ['hardcoded_password', 'hardcoded_secret'] else 'MEDIUM'
                        })
            except Exception as e:
                self.findings.append({
                    'type': 'audit_error',
                    'issue': 'file_read_error',
                    'file': str(file_path),
                    'error': str(e),
                    'severity': 'LOW'
                })

    def audit_configuration_security(self):
        """配置安全审计"""
        config_files = [
            'config/production.yaml',
            'config/development.yaml',
            'config/testing.yaml',
            'docker-compose.yml',
            'Dockerfile'
        ]

        for config_file in config_files:
            file_path = self.app_directory / config_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查敏感信息泄露
                    if re.search(r'password|secret|key', content, re.IGNORECASE):
                        self.findings.append({
                            'type': 'config_security',
                            'issue': 'potential_sensitive_data',
                            'file': str(file_path),
                            'severity': 'HIGH'
                        })

                    # 检查调试模式
                    if re.search(r'debug.*true|debug.*yes', content, re.IGNORECASE):
                        self.findings.append({
                            'type': 'config_security',
                            'issue': 'debug_enabled',
                            'file': str(file_path),
                            'severity': 'MEDIUM'
                        })

                except Exception as e:
                    self.findings.append({
                        'type': 'audit_error',
                        'issue': 'config_read_error',
                        'file': str(file_path),
                        'error': str(e),
                        'severity': 'LOW'
                    })

    def audit_dependencies_security(self):
        """依赖安全审计"""
        requirements_file = self.app_directory / 'requirements.txt'
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r', encoding='utf-8') as f:
                    dependencies = f.read().split('\n')

                # 检查已知的安全问题依赖
                vulnerable_deps = {
                    'django': '3.2.16',  # 假设的版本
                    'flask': '2.2.0',
                    'requests': '2.28.0'
                }

                for dep in dependencies:
                    if dep.strip():
                        dep_name = dep.split('==')[0].split('>=')[0].strip()
                        if dep_name in vulnerable_deps:
                            self.findings.append({
                                'type': 'dependency_security',
                                'issue': 'potentially_vulnerable_dependency',
                                'dependency': dep_name,
                                'severity': 'MEDIUM'
                            })

            except Exception as e:
                self.findings.append({
                    'type': 'audit_error',
                    'issue': 'requirements_read_error',
                    'error': str(e),
                    'severity': 'LOW'
                })

    def generate_audit_report(self):
        """生成审计报告"""
        report = {
            'audit_timestamp': datetime.now().isoformat(),
            'total_findings': len(self.findings),
            'high_severity': len([f for f in self.findings if f['severity'] == 'HIGH']),
            'medium_severity': len([f for f in self.findings if f['severity'] == 'MEDIUM']),
            'low_severity': len([f for f in self.findings if f['severity'] == 'LOW']),
            'findings': self.findings
        }

        # 保存报告
        reports_dir = self.app_directory / 'deploy' / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_file = reports_dir / f'security_audit_report_{int(datetime.now().timestamp())}.json'

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report_file

    def run_audit(self):
        """运行完整审计"""
        print("开始应用安全审计...")

        self.audit_code_security()
        self.audit_configuration_security()
        self.audit_dependencies_security()

        report_file = self.generate_audit_report()

        print(f"审计完成，发现 {len(self.findings)} 个问题")
        print(f"报告已保存到: {report_file}")

        # 按严重程度统计
        severity_count = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for finding in self.findings:
            severity_count[finding['severity']] += 1

        print("问题统计:")
        for severity, count in severity_count.items():
            print(f"  {severity}: {count}")

        return len(self.findings)
```

## 安全事件响应

### 1. 事件分级

#### 1.1 安全事件分级标准

| 级别 | 描述 | 影响范围 | 响应时间 |
|------|------|----------|----------|
| P1 | 严重安全事件 | 系统完全不可用 | 立即响应 |
| P2 | 高危安全事件 | 核心功能受损 | 1小时内响应 |
| P3 | 中危安全事件 | 部分功能受损 | 4小时内响应 |
| P4 | 低危安全事件 | 潜在风险 | 24小时内响应 |

#### 1.2 事件分类

- **入侵事件**: 未经授权的系统访问
- **数据泄露**: 敏感数据外泄
- **服务攻击**: DDoS、暴力破解等
- **漏洞利用**: 利用系统漏洞进行攻击
- **内部威胁**: 内部人员违规操作
- **误操作**: 运维操作失误

### 2. 事件响应流程

#### 2.1 事件检测

```bash
# 1. 监控系统检测
# - Prometheus告警
# - OSSEC入侵检测
# - 应用日志分析

# 2. 安全日志分析
grep -i "error\|fail\|attack" /var/log/rqa2025/security.log

# 3. 系统状态检查
# 检查CPU、内存、磁盘使用率
# 检查网络连接状态
# 检查服务运行状态
```

#### 2.2 事件确认

```bash
# 1. 验证告警真实性
# 检查监控指标
# 验证日志记录
# 确认影响范围

# 2. 收集事件信息
echo "事件确认清单:"
echo "- 检测时间: $(date)"
echo "- 告警来源: "
echo "- 影响系统: "
echo "- 影响范围: "
echo "- 事件描述: "
```

#### 2.3 事件隔离

```bash
# 1. 隔离受影响系统
# 停止受影响的服务
sudo systemctl stop rqa2025

# 2. 网络隔离
# 断开网络连接
sudo ifdown eth0

# 3. 数据隔离
# 停止数据库写入
psql -h localhost -U rqa2025 -d rqa2025 -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'active';"
```

#### 2.4 事件调查

```bash
# 1. 收集系统日志
echo "收集系统日志..."
cp /var/log/syslog /tmp/incident_syslog_$(date +%Y%m%d_%H%M%S)
cp /var/log/auth.log /tmp/incident_auth_$(date +%Y%m%d_%H%M%S)

# 2. 收集应用日志
echo "收集应用日志..."
cp /var/log/rqa2025/*.log /tmp/incident_app_$(date +%Y%m%d_%H%M%S)/

# 3. 收集网络信息
echo "收集网络信息..."
netstat -tlnp > /tmp/incident_netstat_$(date +%Y%m%d_%H%M%S)
iptables -L -n > /tmp/incident_iptables_$(date +%Y%m%d_%H%M%S)

# 4. 收集进程信息
echo "收集进程信息..."
ps aux > /tmp/incident_ps_$(date +%Y%m%d_%H%M%S)
lsof > /tmp/incident_lsof_$(date +%Y%m%d_%H%M%S)
```

#### 2.5 事件修复

```bash
# 1. 修复系统漏洞
# 更新系统补丁
sudo apt-get update && sudo apt-get upgrade

# 2. 修复配置问题
# 恢复安全配置
# 更新防火墙规则

# 3. 清理入侵痕迹
# 删除后门程序
# 清理异常用户
# 重置密码

# 4. 恢复服务
# 启动受影响的服务
sudo systemctl start rqa2025
sudo systemctl start postgresql
sudo systemctl start redis
```

#### 2.6 事件复盘

```markdown
# 安全事件复盘报告

## 事件基本信息
- 事件ID: SEC-2025-001
- 检测时间: 2025-01-15 10:30:00
- 修复时间: 2025-01-15 12:00:00
- 事件等级: P2 (高危)

## 事件描述
检测到来自IP 192.168.1.100的多次登录失败尝试，疑似暴力破解攻击。

## 影响评估
- 影响范围: 认证系统
- 业务影响: 部分用户登录受阻
- 数据影响: 无数据泄露

## 根本原因
SSH服务配置不当，未启用登录失败限制。

## 修复措施
1. 修改SSH配置，启用登录失败限制
2. 增加防火墙规则，限制登录频率
3. 启用双因素认证

## 预防措施
1. 定期安全配置检查
2. 实施自动化安全扫描
3. 加强员工安全意识培训

## 改进计划
- [ ] 实施24小时监控值班
- [ ] 增加安全日志分析频率
- [ ] 优化告警规则准确性
- [ ] 建立安全事件应急预案
```

## 安全合规要求

### 1. 合规性检查清单

#### 1.1 数据保护合规

- [ ] **GDPR合规性**
  - [ ] 数据最小化原则
  - [ ] 个人数据保护
  - [ ] 用户同意管理
  - [ ] 数据访问日志

- [ ] **金融监管合规**
  - [ ] 交易数据记录完整性
  - [ ] 审计日志完整性
  - [ ] 风险控制机制
  - [ ] 业务连续性保证

#### 1.2 安全控制合规

- [ ] **访问控制**
  - [ ] 最小权限原则
  - [ ] 角色分离
  - [ ] 访问审计

- [ ] **加密要求**
  - [ ] 数据传输加密
  - [ ] 数据存储加密
  - [ ] 密钥管理

- [ ] **监控和响应**
  - [ ] 安全事件监控
  - [ ] 事件响应流程
  - [ ] 安全培训

### 2. 合规性审计

#### 2.1 定期安全审计

```bash
#!/bin/bash
# compliance_audit.sh

echo "=== 合规性安全审计 ==="
echo "审计时间: $(date)"

# 1. 检查数据保护措施
echo "1. 数据保护检查"
echo "敏感数据加密状态:"
# 检查数据库加密配置
psql -h localhost -U rqa2025 -d rqa2025 -c "SHOW ssl;" 2>/dev/null || echo "数据库SSL未启用"

echo "备份加密状态:"
# 检查备份文件是否加密
find /data/backup -name "*.enc" | wc -l

# 2. 检查访问控制
echo "2. 访问控制检查"
echo "用户权限配置:"
# 检查用户权限设置
ls -la /etc/rqa2025/

echo "防火墙配置:"
sudo ufw status

# 3. 检查监控和日志
echo "3. 监控和日志检查"
echo "安全日志状态:"
ls -la /var/log/rqa2025/security.log

echo "监控系统状态:"
curl -s http://localhost:9090/-/healthy || echo "Prometheus不可用"

# 4. 检查安全配置
echo "4. 安全配置检查"
echo "SSH安全配置:"
grep -E "^(PermitRootLogin|PasswordAuthentication)" /etc/ssh/sshd_config

echo "系统安全更新:"
apt list --upgradable | grep -i security | wc -l

echo "=== 合规性审计完成 ==="
```

## 总结

安全运维是一个系统性的工程，需要：

1. **建立安全基线**: 通过最小权限原则和访问控制建立安全基础
2. **实施监控告警**: 使用多层次的监控体系及时发现安全威胁
3. **制定响应流程**: 建立完善的安全事件响应和处理流程
4. **持续审计改进**: 通过定期安全审计发现和修复安全问题
5. **合规性保证**: 确保系统符合相关法律法规和行业标准

安全运维不仅需要技术措施，更需要制度保障和人员培训，形成完整的安全生态体系。

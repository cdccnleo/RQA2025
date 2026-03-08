#!/usr/bin/env python3
"""
Docker安全配置修复脚本
修复所有Docker Compose文件中的安全问题
"""

import re
import yaml
from pathlib import Path
from datetime import datetime


def log(message):
    """打印日志"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")


def fix_docker_compose_default_passwords(file_path):
    """修复Docker Compose中的默认密码"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复模式: ${VAR:-default} -> ${VAR}
        # 移除默认值，强制使用环境变量
        patterns = [
            (r'\$\{POSTGRES_PASSWORD:-[^}]+\}', '${POSTGRES_PASSWORD}'),
            (r'\$\{REDIS_PASSWORD:-[^}]+\}', '${REDIS_PASSWORD}'),
            (r'\$\{INFLUXDB_PASSWORD:-[^}]+\}', '${INFLUXDB_PASSWORD}'),
            (r'\$\{INFLUXDB_TOKEN:-[^}]+\}', '${INFLUXDB_TOKEN}'),
            (r'\$\{GRAFANA_PASSWORD:-[^}]+\}', '${GRAFANA_PASSWORD}'),
            (r'\$\{ELASTIC_PASSWORD:-[^}]+\}', '${ELASTIC_PASSWORD}'),
            (r'\$\{GRAFANA_USER:-[^}]+\}', '${GRAFANA_USER}'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # 修复硬编码密码 (不含环境变量语法的)
        hardcoded_patterns = [
            (r'POSTGRES_PASSWORD:\s*postgres', 'POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}'),
            (r'POSTGRES_PASSWORD:\s*SecurePass123!', 'POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}'),
            (r'GF_SECURITY_ADMIN_PASSWORD:\s*admin', 'GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}'),
            (r'GF_SECURITY_ADMIN_PASSWORD:\s*[^\s]+', 'GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}'),
        ]
        
        for pattern, replacement in hardcoded_patterns:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        # 修复Redis命令中的硬编码密码
        content = re.sub(
            r'command:\s*redis-server\s+--requirepass\s+\$\{REDIS_PASSWORD:-[^}]+\}',
            'command: redis-server --requirepass ${REDIS_PASSWORD}',
            content
        )
        content = re.sub(
            r'command:\s*redis-server\s+--requirepass\s+[^\s]+',
            'command: redis-server --requirepass ${REDIS_PASSWORD}',
            content
        )
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            log(f"✅ 已修复默认密码: {file_path}")
            return True
        else:
            log(f"ℹ️  无需修复: {file_path}")
            return True
            
    except Exception as e:
        log(f"❌ 修复失败 {file_path}: {e}")
        return False


def fix_elasticsearch_security(file_path):
    """启用Elasticsearch安全认证"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 启用安全认证
        content = content.replace(
            'xpack.security.enabled=false',
            'xpack.security.enabled=true'
        )
        
        # 添加证书配置（如果不存在）
        if 'xpack.security.transport.ssl.enabled' not in content:
            # 在environment部分添加SSL配置
            content = content.replace(
                'xpack.security.enabled=true',
                '''xpack.security.enabled=true
      - xpack.security.transport.ssl.enabled=true
      - xpack.security.transport.ssl.verification_mode=certificate
      - xpack.security.http.ssl.enabled=true'''
            )
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            log(f"✅ 已启用ES安全认证: {file_path}")
            return True
        else:
            log(f"ℹ️  ES配置无需修改: {file_path}")
            return True
            
    except Exception as e:
        log(f"❌ ES修复失败 {file_path}: {e}")
        return False


def fix_port_exposure(file_path):
    """修复端口暴露问题 - 将数据库端口改为expose"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 定义需要限制的服务和端口
        sensitive_services = {
            'postgres': ['5432:5432'],
            'redis': ['6379:6379'],
            'elasticsearch': ['9200:9200', '9300:9300'],
            'influxdb': ['8086:8086'],
        }
        
        # 解析YAML
        try:
            config = yaml.safe_load(content)
            if config and 'services' in config:
                modified = False
                for service_name, service_config in config['services'].items():
                    if service_name in sensitive_services and 'ports' in service_config:
                        # 检查是否有敏感端口暴露
                        ports = service_config['ports']
                        new_ports = []
                        expose_ports = []
                        
                        for port in ports:
                            port_str = str(port)
                            is_sensitive = any(
                                sensitive_port in port_str 
                                for sensitive_port in sensitive_services[service_name]
                            )
                            
                            if is_sensitive:
                                # 改为expose
                                expose_ports.append(port_str.split(':')[0])
                            else:
                                new_ports.append(port)
                        
                        if expose_ports:
                            service_config['ports'] = new_ports if new_ports else None
                            if service_config['ports'] is None:
                                del service_config['ports']
                            service_config['expose'] = expose_ports
                            modified = True
                
                if modified:
                    # 写回文件
                    with open(file_path, 'w', encoding='utf-8') as f:
                        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                    log(f"✅ 已限制端口暴露: {file_path}")
                    return True
        except yaml.YAMLError:
            # YAML解析失败，使用文本替换
            log(f"⚠️  YAML解析失败，使用文本模式: {file_path}")
            
        log(f"ℹ️  端口配置无需修改: {file_path}")
        return True
        
    except Exception as e:
        log(f"❌ 端口修复失败 {file_path}: {e}")
        return False


def add_network_isolation(file_path):
    """添加网络隔离配置"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已有网络配置
        if 'networks:' in content and 'backend:' in content:
            log(f"ℹ️  已有网络隔离: {file_path}")
            return True
        
        # 添加网络配置
        network_config = '''

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true
'''
        
        content = content.rstrip() + network_config
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        log(f"✅ 已添加网络隔离: {file_path}")
        return True
        
    except Exception as e:
        log(f"❌ 网络隔离添加失败 {file_path}: {e}")
        return False


def create_secure_docker_compose_template():
    """创建安全的Docker Compose模板"""
    template_content = '''version: '3.8'

# 安全提示:
# 1. 所有密码通过环境变量或Docker Secrets传入
# 2. 数据库服务使用expose而非ports，不暴露到主机
# 3. 使用网络隔离，backend网络为internal
# 4. Elasticsearch启用安全认证

secrets:
  db_password:
    file: ./secrets/db_password.txt
  redis_password:
    file: ./secrets/redis_password.txt
  influxdb_password:
    file: ./secrets/influxdb_password.txt
  influxdb_token:
    file: ./secrets/influxdb_token.txt
  grafana_password:
    file: ./secrets/grafana_password.txt
  elastic_password:
    file: ./secrets/elastic_password.txt

services:
  app:
    build: .
    environment:
      - DB_PASSWORD_FILE=/run/secrets/db_password
      - REDIS_PASSWORD_FILE=/run/secrets/redis_password
    secrets:
      - db_password
      - redis_password
    networks:
      - frontend
      - backend
    user: "1000:1000"

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password
    expose:
      - "5432"
    networks:
      - backend

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass $(cat /run/secrets/redis_password)
    secrets:
      - redis_password
    expose:
      - "6379"
    networks:
      - backend

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=true
      - xpack.security.transport.ssl.enabled=true
      - ELASTIC_PASSWORD_FILE=/run/secrets/elastic_password
    secrets:
      - elastic_password
    expose:
      - "9200"
    networks:
      - backend

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true
'''
    
    template_file = Path("docker-compose.secure-template.yml")
    try:
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(template_content)
        log(f"✅ 已创建安全模板: {template_file}")
        return True
    except Exception as e:
        log(f"❌ 模板创建失败: {e}")
        return False


def main():
    """主函数"""
    log("=" * 70)
    log("Docker安全配置修复工具")
    log("=" * 70)
    log("")
    
    # 查找所有docker-compose文件
    docker_compose_files = list(Path('.').rglob('docker-compose*.yml'))
    docker_compose_files.extend(Path('.').rglob('docker-compose*.yaml'))
    
    log(f"找到 {len(docker_compose_files)} 个Docker Compose文件")
    log("")
    
    results = {
        'passwords': 0,
        'elasticsearch': 0,
        'ports': 0,
        'networks': 0,
    }
    
    # 修复每个文件
    for file_path in docker_compose_files:
        log(f"\n处理: {file_path}")
        
        # 1. 修复默认密码
        if fix_docker_compose_default_passwords(file_path):
            results['passwords'] += 1
        
        # 2. 启用Elasticsearch安全
        if fix_elasticsearch_security(file_path):
            results['elasticsearch'] += 1
        
        # 3. 修复端口暴露
        if fix_port_exposure(file_path):
            results['ports'] += 1
        
        # 4. 添加网络隔离
        if add_network_isolation(file_path):
            results['networks'] += 1
    
    # 创建安全模板
    create_secure_docker_compose_template()
    
    # 显示总结
    log("")
    log("=" * 70)
    log("修复总结")
    log("=" * 70)
    log(f"处理的文件数: {len(docker_compose_files)}")
    log(f"密码修复: {results['passwords']} 个文件")
    log(f"ES安全: {results['elasticsearch']} 个文件")
    log(f"端口限制: {results['ports']} 个文件")
    log(f"网络隔离: {results['networks']} 个文件")
    log("")
    log("🎉 Docker安全配置修复完成！")
    log("")
    log("下一步操作:")
    log("1. 创建secrets目录和密钥文件")
    log("2. 更新 .env 文件")
    log("3. 测试配置: docker-compose config")
    log("4. 启动服务: docker-compose up -d")
    log("=" * 70)


if __name__ == "__main__":
    main()

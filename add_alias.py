import re

def add_method_alias():
    with open('src/infrastructure/config/config_manager.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 在__repr__方法后添加update别名
    pattern = r'(def __repr__\(self\):\s*return f"ConfigManager\(env=\{self\._env\}\)")'
    replacement = r'\1\n\n    update = update_config'
    modified_content = re.sub(pattern, replacement, content)

    with open('src/infrastructure/config/config_manager.py', 'w', encoding='utf-8') as f:
        f.write(modified_content)

if __name__ == '__main__':
    add_method_alias()

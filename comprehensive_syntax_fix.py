import re


def fix_common_syntax_issues():
    with open('src/risk/memory_optimizer.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix docstrings with extra quotes (4 quotes -> 3 quotes)
    content = re.sub(r'""".*?"""', lambda m: m.group(0).replace('"""',
                     '"').replace('""', '"'), content)

    # Fix common indentation patterns in method bodies
    lines = content.split('\n')
    fixed_lines = []
    in_method = False
    method_indent = 0

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('def ') or stripped.startswith('class '):
            in_method = True
            method_indent = len(line) - len(stripped)
            fixed_lines.append(line)
        elif stripped and not stripped.startswith(' ') and not stripped.startswith('\t'):
            # This might be a new class/function definition
            in_method = False
            fixed_lines.append(line)
        elif in_method and stripped:
            # Ensure proper indentation for method content
            current_indent = len(line) - len(stripped)
            if current_indent < method_indent + 4:
                # Add proper indentation
                fixed_line = ' ' * (method_indent + 4) + stripped
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    content = '\n'.join(fixed_lines)

    with open('src/risk/memory_optimizer.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Applied comprehensive syntax fixes")


if __name__ == "__main__":
    fix_common_syntax_issues()

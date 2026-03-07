import re


def fix_remaining_syntax_issues():
    with open('src/risk/memory_optimizer.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix docstrings with extra quotes (any number > 3 quotes at start)
    content = re.sub(r'""".*?"""', lambda m: m.group(0).replace('"""',
                     '"').replace('""', '"'), content)

    # Fix method signatures that are split incorrectly
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if this is a method signature that continues on next line
        if ('def ' in line and line.strip().endswith(',')) or ('def ' in line and '(' in line and not line.strip().endswith('):')):
            # Look for continuation lines
            method_lines = [line]
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                method_lines.append(next_line)
                if '):' in next_line or next_line.strip().endswith('):'):
                    break
                j += 1
            # Join and fix the method signature
            method_block = '\n'.join(method_lines)
            method_block = re.sub(r',\s*\n\s*(\w+)', r', \1', method_block)
            method_block = re.sub(r'\(\s*\n\s*', '(', method_block)
            method_block = re.sub(r',\s*\n\s*\)', ')', method_block)
            fixed_lines.append(method_block.replace('\n', ''))
            i = j + 1
        else:
            fixed_lines.append(line)
            i += 1

    content = '\n'.join(fixed_lines)

    # Fix indentation issues in method bodies
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
        elif in_method and stripped and not stripped.startswith('"""'):
            # Ensure proper indentation for method content
            current_indent = len(line) - len(stripped)
            if current_indent < method_indent + 4 and not stripped.startswith('#'):
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

    print("Applied final comprehensive syntax fixes")


if __name__ == "__main__":
    fix_remaining_syntax_issues()

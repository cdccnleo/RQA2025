with open('src/risk/memory_optimizer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Split content into lines and reconstruct properly
lines = content.split('\n')
fixed_lines = []

for line in lines:
    # Handle merged lines that contain multiple statements
    if '@dataclass' in line and 'class' in line and ':' in line:
        # Split merged @dataclassclass declarations
        parts = line.split('@dataclassclass')
        for i, part in enumerate(parts):
            if part.strip():
                if i == 0:
                    fixed_lines.append('@dataclass')
                fixed_lines.append(f'class{part}')
                if i < len(parts) - 1:
                    fixed_lines.append('')

    elif 'class MemoryMonitor:' in line and 'def __init__' in line:
        # Split merged class and method declarations
        fixed_lines.append('class MemoryMonitor:')
        fixed_lines.append('    """内存监控器"""')
        fixed_lines.append('')
        fixed_lines.append(
            '    def __init__(self, check_interval: int = 30, enable_tracemalloc: bool = True):')

    elif line.strip() and not line.startswith(' ') and not line.startswith('\t') and len(line.strip().split()) > 5:
        # This might be a merged line - try to split it
        words = line.strip().split()
        if len(words) > 10:  # Likely a merged line
            # Try to identify logical breaks
            if 'class' in words and ':' in line:
                # This is a class declaration
                class_idx = words.index('class')
                fixed_lines.append(' '.join(words[class_idx:class_idx+3]))
                remaining = ' '.join(words[class_idx+3:])
                if remaining:
                    fixed_lines.append('    ' + remaining)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)

# Write back with proper formatting
content = '\n'.join(fixed_lines)
with open('src/risk/memory_optimizer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Applied comprehensive reconstruction')

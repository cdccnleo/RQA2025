#!/usr/bin/env python3
"""
自动应用常量替换魔数

批量替换代码中的魔数为统一的常量定义
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict


class ConstantsReplacementTool:
    """常量替换工具"""
    
    def __init__(self):
        """初始化"""
        self.replacements_made = 0
        self.files_modified = 0
    
    def replace_http_status_codes(self, file_path: Path) -> int:
        """替换HTTP状态码"""
        replacements = {
            r'\b200\b': 'HTTPConstants.OK',
            r'\b201\b': 'HTTPConstants.CREATED',
            r'\b400\b': 'HTTPConstants.BAD_REQUEST',
            r'\b401\b': 'HTTPConstants.UNAUTHORIZED',
            r'\b403\b': 'HTTPConstants.FORBIDDEN',
            r'\b404\b': 'HTTPConstants.NOT_FOUND',
            r'\b500\b': 'HTTPConstants.INTERNAL_SERVER_ERROR',
        }
        
        return self._apply_replacements(file_path, replacements, 
                                       "from src.infrastructure.constants import HTTPConstants")
    
    def replace_time_constants(self, file_path: Path) -> int:
        """替换时间常量"""
        replacements = {
            r'\b30\b(?=\s*#.*秒|.*timeout|.*interval)': 'TimeConstants.TIMEOUT_NORMAL',
            r'\b60\b(?=\s*#.*分钟|.*minute)': 'TimeConstants.MINUTE',
            r'\b3600\b(?=\s*#.*小时|.*hour)': 'TimeConstants.HOUR',
            r'\b86400\b(?=\s*#.*天|.*day)': 'TimeConstants.DAY',
            r'\b300\b(?=\s*#.*分钟|.*interval)': 'TimeConstants.MONITOR_INTERVAL_VERY_SLOW',
        }
        
        return self._apply_replacements(file_path, replacements,
                                       "from src.infrastructure.constants import TimeConstants")
    
    def replace_size_constants(self, file_path: Path) -> int:
        """替换大小常量"""
        replacements = {
            r'\b1024\b(?=\s*#.*KB|.*kilobyte)': 'SizeConstants.KB',
            r'/\s*1024\s*/\s*1024': '/ SizeConstants.MB',
        }
        
        return self._apply_replacements(file_path, replacements,
                                       "from src.infrastructure.constants import SizeConstants")
    
    def _apply_replacements(self, file_path: Path, replacements: Dict[str, str], 
                          import_statement: str) -> int:
        """应用替换"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            count = 0
            
            # 应用所有替换
            for pattern, replacement in replacements.items():
                matches = re.findall(pattern, content)
                if matches:
                    content = re.sub(pattern, replacement, content)
                    count += len(matches)
            
            # 如果有替换，添加import语句
            if count > 0 and import_statement not in content:
                # 在文件顶部添加import
                lines = content.split('\n')
                
                # 找到合适的插入位置（在其他import之后）
                insert_pos = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        insert_pos = i + 1
                    elif insert_pos > 0 and not line.strip().startswith('import ') and not line.strip().startswith('from '):
                        break
                
                lines.insert(insert_pos, import_statement)
                content = '\n'.join(lines)
            
            # 只有在有变化时才写回
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.files_modified += 1
                self.replacements_made += count
                print(f"  ✅ {file_path.name}: {count}处替换")
                return count
            
            return 0
            
        except Exception as e:
            print(f"  ⚠️  {file_path.name}: 替换失败 - {e}")
            return 0
    
    def process_module(self, module_path: Path):
        """处理整个模块"""
        print(f"\n处理模块: {module_path}")
        
        py_files = list(module_path.rglob("*.py"))
        print(f"找到{len(py_files)}个Python文件")
        
        for py_file in py_files:
            # 跳过__pycache__等
            if '__pycache__' in str(py_file) or 'test_' in py_file.name:
                continue
            
            # 替换HTTP状态码
            self.replace_http_status_codes(py_file)
            
            # 替换时间常量
            self.replace_time_constants(py_file)
            
            # 替换大小常量
            self.replace_size_constants(py_file)


def main():
    """主函数"""
    print("🔧 常量替换工具")
    print("=" * 80)
    
    tool = ConstantsReplacementTool()
    
    # 处理versioning模块（示例）
    versioning_path = Path("src/infrastructure/versioning")
    if versioning_path.exists():
        tool.process_module(versioning_path)
    
    # 处理ops模块
    ops_path = Path("src/infrastructure/ops")
    if ops_path.exists():
        tool.process_module(ops_path)
    
    # 处理distributed模块
    distributed_path = Path("src/infrastructure/distributed")
    if distributed_path.exists():
        tool.process_module(distributed_path)
    
    print("\n" + "=" * 80)
    print(f"✅ 替换完成")
    print(f"   修改文件: {tool.files_modified}个")
    print(f"   替换次数: {tool.replacements_made}处")
    print("=" * 80)


if __name__ == '__main__':
    main()


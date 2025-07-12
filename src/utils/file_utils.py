import os
import tempfile
from typing import Union, IO
from pathlib import Path

def safe_file_write(
    file_path: Union[str, Path],
    content: str,
    mode: str = 'w',
    encoding: str = 'utf-8',
    overwrite: bool = True
) -> None:
    """安全写入文件，确保原子性和错误处理

    参数:
        file_path: 目标文件路径
        content: 要写入的内容
        mode: 文件打开模式(默认'w')
        encoding: 文件编码(默认'utf-8')
        overwrite: 是否覆盖已存在文件(默认True)

    异常:
        IOError: 当文件操作失败时抛出
        ValueError: 当参数无效时抛出
    """
    if not file_path:
        raise ValueError("文件路径不能为空")

    file_path = Path(file_path)
    if file_path.exists() and not overwrite:
        raise IOError(f"文件已存在且不允许覆盖: {file_path}")

    # 创建临时文件
    temp_dir = file_path.parent
    try:
        with tempfile.NamedTemporaryFile(
            mode=mode,
            encoding=encoding,
            dir=str(temp_dir),
            delete=False
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        # 原子性重命名
        if os.name == 'nt':  # Windows
            if file_path.exists():
                os.unlink(file_path)
            os.rename(temp_path, file_path)
        else:  # Unix-like
            os.replace(temp_path, file_path)

    except Exception as e:
        # 清理临时文件
        if 'temp_path' in locals() and temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        raise IOError(f"文件写入失败: {str(e)}") from e

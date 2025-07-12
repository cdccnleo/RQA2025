import os
import sys

file_path = r"C:\PythonProject\RQA2025\src\data\performance\preloader.py"

try:
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Successfully deleted {file_path}")
    else:
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"Error deleting file: {str(e)}", file=sys.stderr)
    sys.exit(1)

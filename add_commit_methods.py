#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 读取文件
with open('src/infrastructure/utils/adapters/postgresql_adapter.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 在第410行之后（close方法之前）插入commit和rollback方法
new_lines = [
    "\n",
    "    def commit(self) -> bool:\n",
    "        \"\"\"提交事务\"\"\"\n",
    "        try:\n",
    "            if self._client:\n",
    "                self._client.commit()\n",
    "            return True\n",
    "        except Exception as e:\n",
    "            self._error_handler.handle(e, \"PostgreSQL提交失败\")\n",
    "            return False\n",
    "\n",
    "    def rollback(self) -> bool:\n",
    "        \"\"\"回滚事务\"\"\"\n",
    "        try:\n",
    "            if self._client:\n",
    "                self._client.rollback()\n",
    "            return True\n",
    "        except Exception as e:\n",
    "            self._error_handler.handle(e, \"PostgreSQL回滚失败\")\n",
    "            return False\n",
]

# 在第411行（索引410）之后插入
updated_lines = lines[:411] + new_lines + lines[411:]

# 写回文件
with open('src/infrastructure/utils/adapters/postgresql_adapter.py', 'w', encoding='utf-8') as f:
    f.writelines(updated_lines)

print("Added commit and rollback methods successfully")


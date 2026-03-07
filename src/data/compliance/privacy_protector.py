from typing import Any
import hashlib
import re


class PrivacyProtector:

    """隐私保护"""

    def protect(self, data: Any, level: str = "standard") -> Any:
        """对数据进行隐私保护"""
        if not isinstance(data, str):
            return data

        if not level or level not in ["standard", "encrypted", "none"]:
            level = "standard"

        if level == "none":
            return data
        elif level == "encrypted":
            return hashlib.sha256(data.encode()).hexdigest()
        elif level == "standard":
            return self._mask_data(data)

        return data

    def _mask_data(self, data: str) -> str:
        """对数据进行脱敏处理"""
        if not data:
            return data

        # 手机号模式
        if re.match(r'^1[3-9]\d{9}$', data):
            return data[:2] + "*******" + data[-2:]

        # 邮箱模式
        if '@' in data:
            parts = data.split('@')
            username = parts[0]
            domain = parts[1]
            if len(username) > 1:
                masked_username = username[0] + "***"
            else:
                masked_username = "***"
            return masked_username + "@***." + domain.split('.')[-1]

        # 身份证号模式
        if re.match(r'^\d{17}[\dXx]$', data):
            return data[:6] + "****" + data[-4:]

        # 信用卡号模式
        if re.match(r'^\d{16}$', data):
            return data[:4] + "****" + data[-4:]

        # 银行账号模式
        if re.match(r'^\d{16,19}$', data):
            return data[:4] + "****" + data[-4:]

        # 地址模式
        if len(data) > 8:
            return data[:2] + "****" + data[-2:]

        # 姓名模式
        if len(data) == 2:
            return data[0] + "*"

        # 默认模式：短字符串全部脱敏
        if len(data) <= 4:
            return "*" * len(data)

        # 长字符串：显示前2后2位
        return data[:2] + "****" + data[-2:]

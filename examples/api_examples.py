"""
RQA2025 API 示例代码

提供完整的API调用示例，包括认证、策略管理、回测执行等功能。

作者: 王芳
创建日期: 2026-03-29
版本: 1.0.0
"""

import requests
import time
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class APIConfig:
    """API配置类"""
    base_url: str = "https://api.rqa2025.com/v1"
    timeout: int = 30
    max_retries: int = 3


class RQA2025APIClient:
    """
    RQA2025 API客户端
    
    提供完整的API调用封装，包括：
    - 认证管理
    - 策略管理
    - 特征管理
    - 模型管理
    - 交易执行
    - 监控指标
    
    Example:
        >>> client = RQA2025APIClient()
        >>> client.login("username", "password")
        >>> strategy = client.create_strategy("双均线策略", "...", "technical", {...})
        >>> print(strategy)
    """
    
    def __init__(self, config: Optional[APIConfig] = None):
        """
        初始化API客户端
        
        Args:
            config: API配置，如未提供则使用默认配置
        """
        self.config = config or APIConfig()
        self.token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expires_at: float = 0
        self.session = requests.Session()
        
    def _get_headers(self) -> Dict[str, str]:
        """
        获取请求头
        
        Returns:
            包含认证信息的请求头字典
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "RQA2025-Python-Client/1.0.0"
        }
        
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
            
        return headers
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        执行HTTP请求
        
        Args:
            method: HTTP方法
            endpoint: API端点
            data: 请求体数据
            params: URL参数
            retry_count: 当前重试次数
            
        Returns:
            API响应数据
            
        Raises:
            requests.exceptions.RequestException: 请求失败
        """
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=self._get_headers(),
                json=data,
                params=params,
                timeout=self.config.timeout
            )
            
            # 处理Token过期
            if response.status_code == 401 and retry_count < self.config.max_retries:
                if self._refresh_access_token():
                    return self._make_request(method, endpoint, data, params, retry_count + 1)
            
            # 处理限流
            if response.status_code == 429 and retry_count < self.config.max_retries:
                wait_time = (2 ** retry_count) + 0.5
                print(f"⏳ 遇到限流，等待 {wait_time:.2f} 秒后重试...")
                time.sleep(wait_time)
                return self._make_request(method, endpoint, data, params, retry_count + 1)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if retry_count < self.config.max_retries:
                wait_time = 2 ** retry_count
                print(f"⏳ 请求失败，{wait_time}秒后重试...")
                time.sleep(wait_time)
                return self._make_request(method, endpoint, data, params, retry_count + 1)
            raise
    
    def login(self, username: str, password: str) -> bool:
        """
        用户登录
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            登录是否成功
        """
        try:
            response = self._make_request(
                "POST",
                "/auth/login",
                data={"username": username, "password": password}
            )
            
            if response.get("success"):
                data = response["data"]
                self.token = data["access_token"]
                self.refresh_token = data.get("refresh_token")
                self.token_expires_at = time.time() + data.get("expires_in", 3600)
                print(f"✅ 登录成功！Token有效期: {data.get('expires_in', 3600)}秒")
                return True
            else:
                print(f"❌ 登录失败: {response.get('error', {}).get('message', '未知错误')}")
                return False
                
        except Exception as e:
            print(f"❌ 登录失败: {e}")
            return False
    
    def _refresh_access_token(self) -> bool:
        """
        刷新访问令牌
        
        Returns:
            刷新是否成功
        """
        if not self.refresh_token:
            return False
            
        try:
            response = self._make_request(
                "POST",
                "/auth/refresh",
                data={"refresh_token": self.refresh_token}
            )
            
            if response.get("success"):
                data = response["data"]
                self.token = data["access_token"]
                self.token_expires_at = time.time() + data.get("expires_in", 3600)
                print("✅ Token刷新成功")
                return True
                
        except Exception as e:
            print(f"❌ Token刷新失败: {e}")
            
        return False
    
    # ==================== 策略管理 ====================
    
    def create_strategy(
        self,
        name: str,
        description: str,
        strategy_type: str,
        parameters: Dict[str, Any],
        code: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        创建策略
        
        Args:
            name: 策略名称
            description: 策略描述
            strategy_type: 策略类型 (technical/fundamental/ml)
            parameters: 策略参数
            code: 策略代码（可选）
            
        Returns:
            创建的策略信息
        """
        data = {
            "name": name,
            "description": description,
            "type": strategy_type,
            "parameters": parameters
        }
        
        if code:
            data["code"] = code
            
        try:
            response = self._make_request("POST", "/strategies", data=data)
            if response.get("success"):
                strategy = response["data"]
                print(f"✅ 策略创建成功: {strategy['id']}")
                return strategy
            else:
                print(f"❌ 策略创建失败: {response.get('error', {}).get('message')}")
                return None
        except Exception as e:
            print(f"❌ 策略创建失败: {e}")
            return None
    
    def get_strategies(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        strategy_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取策略列表
        
        Args:
            page: 页码
            page_size: 每页数量
            status: 状态筛选
            strategy_type: 类型筛选
            
        Returns:
            策略列表和分页信息
        """
        params = {"page": page, "page_size": page_size}
        
        if status:
            params["status"] = status
        if strategy_type:
            params["type"] = strategy_type
            
        try:
            response = self._make_request("GET", "/strategies", params=params)
            return response.get("data", {})
        except Exception as e:
            print(f"❌ 获取策略列表失败: {e}")
            return {"items": [], "pagination": {}}
    
    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        获取策略详情
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            策略详细信息
        """
        try:
            response = self._make_request("GET", f"/strategies/{strategy_id}")
            return response.get("data") if response.get("success") else None
        except Exception as e:
            print(f"❌ 获取策略详情失败: {e}")
            return None
    
    def update_strategy(
        self,
        strategy_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        更新策略
        
        Args:
            strategy_id: 策略ID
            name: 新名称
            description: 新描述
            parameters: 新参数
            
        Returns:
            更新后的策略信息
        """
        data = {}
        if name:
            data["name"] = name
        if description:
            data["description"] = description
        if parameters:
            data["parameters"] = parameters
            
        try:
            response = self._make_request("PUT", f"/strategies/{strategy_id}", data=data)
            if response.get("success"):
                print(f"✅ 策略更新成功: {strategy_id}")
                return response.get("data")
            else:
                print(f"❌ 策略更新失败: {response.get('error', {}).get('message')}")
                return None
        except Exception as e:
            print(f"❌ 策略更新失败: {e}")
            return None
    
    def delete_strategy(self, strategy_id: str) -> bool:
        """
        删除策略
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            删除是否成功
        """
        try:
            response = self._make_request("DELETE", f"/strategies/{strategy_id}")
            if response.get("success"):
                print(f"✅ 策略删除成功: {strategy_id}")
                return True
            else:
                print(f"❌ 策略删除失败: {response.get('error', {}).get('message')}")
                return False
        except Exception as e:
            print(f"❌ 策略删除失败: {e}")
            return False
    
    def run_backtest(
        self,
        strategy_id: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000,
        commission: float = 0.001
    ) -> Optional[Dict[str, Any]]:
        """
        执行策略回测
        
        Args:
            strategy_id: 策略ID
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            initial_capital: 初始资金
            commission: 手续费率
            
        Returns:
            回测任务信息
        """
        data = {
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "commission": commission
        }
        
        try:
            response = self._make_request(
                "POST",
                f"/strategies/{strategy_id}/backtest",
                data=data
            )
            
            if response.get("success"):
                backtest = response["data"]
                print(f"✅ 回测启动成功: {backtest['backtest_id']}")
                return backtest
            else:
                print(f"❌ 回测启动失败: {response.get('error', {}).get('message')}")
                return None
        except Exception as e:
            print(f"❌ 回测启动失败: {e}")
            return None
    
    def get_backtest_result(self, backtest_id: str) -> Optional[Dict[str, Any]]:
        """
        获取回测结果
        
        Args:
            backtest_id: 回测ID
            
        Returns:
            回测结果
        """
        try:
            response = self._make_request("GET", f"/backtests/{backtest_id}")
            return response.get("data") if response.get("success") else None
        except Exception as e:
            print(f"❌ 获取回测结果失败: {e}")
            return None
    
    def wait_for_backtest(
        self,
        backtest_id: str,
        timeout: int = 300,
        poll_interval: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        等待回测完成
        
        Args:
            backtest_id: 回测ID
            timeout: 超时时间（秒）
            poll_interval: 轮询间隔（秒）
            
        Returns:
            回测结果
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = self.get_backtest_result(backtest_id)
            
            if not result:
                print("❌ 获取回测结果失败")
                return None
            
            status = result.get("status")
            
            if status == "completed":
                print("✅ 回测完成！")
                performance = result.get("performance", {})
                print(f"📊 总收益率: {performance.get('total_return', 0)}%")
                print(f"📊 夏普比率: {performance.get('sharpe_ratio', 0)}")
                print(f"📊 最大回撤: {performance.get('max_drawdown', 0)}%")
                return result
            elif status == "failed":
                print(f"❌ 回测失败: {result.get('error', '未知错误')}")
                return None
            else:
                progress = result.get("progress", 0)
                print(f"⏳ 回测进行中... {progress}%")
            
            time.sleep(poll_interval)
        
        print("⏰ 等待超时")
        return None


# ==================== 使用示例 ====================

def example_create_and_backtest_strategy():
    """示例：创建策略并执行回测"""
    
    # 初始化客户端
    client = RQA2025APIClient()
    
    # 登录
    if not client.login("your_username", "your_password"):
        print("❌ 登录失败")
        return
    
    # 创建策略
    strategy = client.create_strategy(
        name="双均线策略",
        description="基于5日和20日均线的交叉策略",
        strategy_type="technical",
        parameters={
            "short_period": 5,
            "long_period": 20,
            "symbol": "000001.SZ"
        }
    )
    
    if not strategy:
        return
    
    # 执行回测
    backtest = client.run_backtest(
        strategy_id=strategy["id"],
        start_date="2025-01-01",
        end_date="2025-12-31",
        initial_capital=100000
    )
    
    if not backtest:
        return
    
    # 等待回测完成
    result = client.wait_for_backtest(backtest["backtest_id"])
    
    if result:
        print("\n🎉 策略回测完成！")
        print(json.dumps(result, indent=2, ensure_ascii=False))


def example_list_and_manage_strategies():
    """示例：列出和管理策略"""
    
    client = RQA2025APIClient()
    
    if not client.login("your_username", "your_password"):
        return
    
    # 获取策略列表
    strategies = client.get_strategies(page=1, page_size=10)
    
    print(f"📋 共找到 {strategies.get('pagination', {}).get('total', 0)} 个策略")
    
    for strategy in strategies.get("items", []):
        print(f"  - {strategy['name']} ({strategy['id']}) - {strategy['status']}")
    
    # 获取第一个策略的详情
    if strategies.get("items"):
        first_strategy = strategies["items"][0]
        detail = client.get_strategy(first_strategy["id"])
        
        if detail:
            print(f"\n📖 策略详情: {detail['name']}")
            print(f"   描述: {detail['description']}")
            print(f"   参数: {json.dumps(detail['parameters'], indent=2)}")


def example_error_handling():
    """示例：错误处理"""
    
    client = RQA2025APIClient()
    
    # 错误的登录凭证
    success = client.login("wrong_username", "wrong_password")
    
    if not success:
        print("⚠️ 登录失败，请检查用户名和密码")
        return
    
    # 尝试获取不存在的策略
    strategy = client.get_strategy("non_existent_id")
    
    if strategy is None:
        print("⚠️ 策略不存在")


if __name__ == "__main__":
    print("=" * 60)
    print("RQA2025 API 示例代码")
    print("=" * 60)
    
    # 运行示例
    # example_create_and_backtest_strategy()
    # example_list_and_manage_strategies()
    # example_error_handling()
    
    print("\n💡 提示: 取消注释上面的函数调用来运行示例")
    print("📖 请确保已配置正确的API凭证")

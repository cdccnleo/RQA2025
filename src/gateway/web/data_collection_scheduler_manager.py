"""
鏁版嵁閲囬泦璋冨害绠＄悊鍣�

鏍规嵁鏁版嵁婧愮殑閲囬泦棰戠巼閰嶇疆锛屽畾鏃舵��鏌ュ苟鑷�鍔ㄧ敓鎴愰噰闆嗕换鍔°��
"""

import threading
import logging
import time
from typing import Dict, Any, Optional, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from src.gateway.web.rate_limit_parser import should_collect, parse_rate_limit
from src.core.orchestration.scheduler import (
    get_unified_scheduler, TaskType, TaskPriority
)

logger = logging.getLogger(__name__)


class DataCollectionSchedulerManager:
    """
    鏁版嵁閲囬泦璋冨害绠＄悊鍣�
    
    瀹氭椂妫�鏌ュ凡鍚�鐢ㄧ殑鏁版嵁婧愶紝鏍规嵁閲囬泦棰戠巼鑷�鍔ㄧ敓鎴愰噰闆嗕换鍔°��
    """
    
    def __init__(self, check_interval: int = 10):
        """
        鍒濆�嬪寲璋冨害绠＄悊鍣�
        
        Args:
            check_interval: 妫�鏌ラ棿闅旓紙绉掞級锛岄粯璁�10绉掞紙缂╃煭妫�鏌ラ棿闅斾互鎻愰珮鍝嶅簲鎬э級
        """
        self._running = False
        self._check_interval = check_interval
        self._scheduler_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # 璺熻釜宸叉彁浜ょ殑浠诲姟锛岄伩鍏嶉噸澶�
        self._submitted_tasks: Set[str] = set()
        self._completion_timestamps: Dict[str, str] = {}  # 淇�澶岯ug: 淇濆瓨鎻愪氦鏃剁殑鏃堕棿鎴筹紝閬垮厤鍥炶皟鏃舵椂闂存埑涓嶄竴鑷�
        
        # 缁熻�′俊鎭�
        self._stats = {
            "total_checks": 0,
            "tasks_submitted": 0,
            "sources_checked": 0,
            "last_check_time": None,
            "next_check_time": None
        }
        
        logger.info(f"鏁版嵁閲囬泦璋冨害绠＄悊鍣ㄥ垵濮嬪寲瀹屾垚锛屾��鏌ラ棿闅�: {check_interval}绉�")
    
    def start(self) -> bool:
        """
        鍚�鍔ㄨ皟搴︾�＄悊鍣�
        
        Returns:
            bool: 鏄�鍚︽垚鍔熷惎鍔�
        """
        with self._lock:
            if self._running:
                logger.debug("璋冨害绠＄悊鍣ㄥ凡鍦ㄨ繍琛屼腑")
                return True
            
            self._running = True
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                name="DataCollectionScheduler",
                daemon=True
            )
            self._scheduler_thread.start()
            
            logger.info("鉁� 鏁版嵁閲囬泦璋冨害绠＄悊鍣ㄥ凡鍚�鍔�")
            return True
    
    def stop(self) -> bool:
        """
        鍋滄�㈣皟搴︾�＄悊鍣�
        
        Returns:
            bool: 鏄�鍚︽垚鍔熷仠姝�
        """
        with self._lock:
            if not self._running:
                logger.debug("璋冨害绠＄悊鍣ㄦ湭鍦ㄨ繍琛�")
                return True
            
            self._running = False
            
            if self._scheduler_thread:
                self._scheduler_thread.join(timeout=5)
            
            logger.info("馃洃 鏁版嵁閲囬泦璋冨害绠＄悊鍣ㄥ凡鍋滄��")
            return True
    
    def _scheduler_loop(self):
        """璋冨害鍣ㄤ富寰�鐜�"""
        logger.info("馃攧 鏁版嵁閲囬泦璋冨害涓诲惊鐜�宸插惎鍔�")
        
        while self._running:
            try:
                # 璁板綍鏈�娆℃��鏌ユ椂闂�
                self._stats["last_check_time"] = datetime.now().isoformat()
                self._stats["next_check_time"] = None
                
                # 鎵ц�屾��鏌ュ拰璋冨害
                self._check_and_schedule()
                
                # 璁＄畻涓嬫�℃��鏌ユ椂闂�
                next_check = datetime.now().timestamp() + self._check_interval
                self._stats["next_check_time"] = datetime.fromtimestamp(next_check).isoformat()
                
                # 浼戠湢鐩村埌涓嬫�℃��鏌�
                for _ in range(self._check_interval):
                    if not self._running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"鉂� 璋冨害寰�鐜�閿欒��: {e}")
                time.sleep(5)
        
        logger.info("馃洃 鏁版嵁閲囬泦璋冨害涓诲惊鐜�宸插仠姝�")
    
    def _check_and_schedule(self):
        """
        妫�鏌ユ暟鎹�婧愬苟鐢熸垚閲囬泦浠诲姟
        """
        logger.debug("馃攳 寮�濮嬫��鏌ユ暟鎹�婧�...")
        
        try:
            # 鑾峰彇鎵�鏈夊凡鍚�鐢ㄧ殑鏁版嵁婧�
            from src.gateway.web.data_source_config_manager import get_data_source_config_manager
            
            config_manager = get_data_source_config_manager()
            sources = config_manager.get_data_sources()
            
            enabled_sources = [s for s in sources if s.get("enabled", False)]
            
            self._stats["sources_checked"] = len(enabled_sources)
            self._stats["total_checks"] += 1
            
            logger.info(f"馃搳 妫�鏌� {len(enabled_sources)} 涓�宸插惎鐢ㄧ殑鏁版嵁婧�")
            
            # 妫�鏌ユ瘡涓�鏁版嵁婧�
            for source in enabled_sources:
                try:
                    self._check_source(source)
                except Exception as e:
                    logger.error(f"妫�鏌ユ暟鎹�婧愬け璐� {source.get('id')}: {e}")
            
            # 娓呯悊宸插畬鎴愮殑浠诲姟璁板綍
            self._cleanup_completed_tasks()
            
        except Exception as e:
            logger.error(f"妫�鏌ユ暟鎹�婧愭椂鍑洪敊: {e}")
    
    def _check_source(self, source: Dict[str, Any]):
        """
        妫�鏌ュ崟涓�鏁版嵁婧愭槸鍚﹂渶瑕侀噰闆�
        
        Args:
            source: 鏁版嵁婧愰厤缃�
        """
        source_id = source.get("id")
        rate_limit = source.get("rate_limit", "")
        
        if not rate_limit:
            logger.debug(f"鏁版嵁婧� {source_id} 娌℃湁閰嶇疆閲囬泦棰戠巼锛岃烦杩�")
            return

        # 2026-04-13 淇�澶�: 浣跨敤 last_collection_time 鑰岄潪 last_test
        # 鍘熷洜: last_test 鐢卞仴搴锋��娴嬫洿鏂帮紙棰戠箒锛夛紝瀵艰嚧 should_collect 閿欒��鍦拌�や负浠婂ぉ宸查噰闆�
        # 鑰� last_collection_time 鎵嶇湡姝ｅ弽鏄犳暟鎹�閲囬泦鏃堕棿
        try:
            from src.gateway.web.data_source_config_manager import get_data_source_config_manager
            config_manager = get_data_source_config_manager()
            fresh_source = config_manager.get_data_source(source_id)
            if fresh_source:
                # 浼樺厛浣跨敤 last_collection_time锛堢湡姝ｅ弽鏄犻噰闆嗘椂闂达級
                last_collection = (
                    fresh_source.get("last_collection") or
                    fresh_source.get("last_collection_time") or
                    fresh_source.get("last_test")  # 闄嶇骇鍒� last_test
                )
                logger.debug(f"鏁版嵁婧� {source_id} 浠庢暟鎹�搴撹幏鍙� last_collection: {last_collection}")
            else:
                last_collection = (
                    source.get("last_collection") or
                    source.get("last_collection_time") or
                    source.get("last_test")
                )
                logger.warning(f"鏁版嵁婧� {source_id} 鏃犳硶浠庢暟鎹�搴撹幏鍙栵紝浣跨敤浼犲叆鐨� last_collection: {last_collection}")
        except Exception as e:
            last_collection = source.get("last_collection") or source.get("last_test")
            logger.error(f"浠庢暟鎹�搴撹幏鍙� last_collection 澶辫触: {e}")
            last_collection = None  # Safety fallback

        # 妫�鏌ユ槸鍚﹀簲璇ラ噰闆�
        if should_collect(last_collection, rate_limit):
            logger.info(f"馃幆 鏁版嵁婧� {source_id} 鍒拌揪閲囬泦鏃堕棿锛坙ast_collection={last_collection}锛夛紝鍑嗗�囨彁浜や换鍔�")
            
            # 鍐嶆�℃��鏌ユ槸鍚﹀凡鏈夊緟澶勭悊鐨勪换鍔★紙鍙岄噸妫�鏌ワ級
            if self._has_pending_task(source_id):
                logger.info(f"馃搮 鏁版嵁婧� {source_id} 浠婂ぉ宸查噰闆嗭紝璺宠繃")
                return
            
            # 鎻愪氦閲囬泦浠诲姟
            self._submit_collection_task(source_id, source)
        else:
            logger.debug(f"鏁版嵁婧� {source_id} 鏈�鍒拌揪閲囬泦鏃堕棿 (last_collection: {last_collection})")
    
    def _has_pending_task(self, source_id: str) -> bool:
        """
        妫�鏌ヤ粖澶╂槸鍚﹀凡閲囬泦杩囨暟鎹�
        浼樺厛鏌ヨ�㈡暟鎹�搴撲腑鐨� last_test 瀛楁�碉紝閬垮厤瀹瑰櫒閲嶅惎鍚庨噸澶嶉噰闆�
        
        Args:
            source_id: 鏁版嵁婧怚D
            
        Returns:
            bool: 浠婂ぉ鏄�鍚﹀凡閲囬泦
        """
        try:
            # 浠庢暟鎹�搴撹幏鍙栨暟鎹�婧愰厤缃�锛堟寔涔呭寲妫�鏌ワ紝閬垮厤瀹瑰櫒閲嶅惎鍚庨噸澶嶉噰闆嗭級
            from src.gateway.web.data_source_config_manager import get_data_source_config_manager
            
            config_manager = get_data_source_config_manager()
            source_config = config_manager.get_data_source(source_id)
            
            if source_config:
                # 2026-04-13 淇�澶�: 浣跨敤 last_collection 鑰岄潪 last_test
                last_collected = (
                    source_config.get("last_collection") or
                    source_config.get("last_collection_time") or
                    source_config.get("last_test")  # 闄嶇骇
                )
                if last_collected:
                    try:
                        if isinstance(last_collected, str):
                            # 灏濊瘯澶氱�� datetime 鏍煎紡
                            _parsed_date = None
                            for _fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                                         "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                                try:
                                    _parsed_date = datetime.strptime(last_collected, _fmt).date()
                                    break
                                except ValueError:
                                    continue
                            last_collected_date = _parsed_date
                        elif isinstance(last_collected, datetime):
                            last_collected_date = last_collected.date()
                        else:
                            logger.warning(f"鏈�鐭ョ殑last_collected绫诲瀷: {type(last_collected)}, 鍊�: {last_collected}")
                            last_collected_date = None

                        if last_collected_date:
                            today = datetime.now().date()

                            if last_collected_date == today:
                                logger.info(f"馃搮 鏁版嵁婧� {source_id} 浠婂ぉ宸查噰闆嗭紙last_collected: {last_collected}锛夛紝璺宠繃")
                                return True
                    except (ValueError, TypeError) as e:
                        logger.warning(f"瑙ｆ瀽 last_collected 澶辫触: {last_collected}, 绫诲瀷: {type(last_collected)}, 閿欒��: {e}")
            
            # 鏁版嵁搴撴��鏌ュけ璐ユ垨鏈�鎵惧埌閰嶇疆锛岄檷绾у埌鍐呭瓨妫�鏌�
            logger.debug(f"鏁版嵁搴撴��鏌ュけ璐ワ紝闄嶇骇鍒板唴瀛樻��鏌�: {source_id}")
            
        except Exception as e:
            logger.error(f"妫�鏌ユ暟鎹�搴撳け璐�: {e}")
        
        # 鍐呭瓨妫�鏌ワ紙浣滀负缂撳瓨鍜岄檷绾ф柟妗堬級
        task_key = f"{source_id}:{datetime.now().strftime('%Y%m%d')}"
        if task_key in self._submitted_tasks:
            logger.debug(f"鍐呭瓨妫�鏌ワ細鏁版嵁婧� {source_id} 浠婂ぉ宸叉彁浜よ繃浠诲姟")
            return True
        
        return False
    
    def _submit_collection_task(self, source_id: str, source_config: Dict[str, Any]):
        """
        鎻愪氦閲囬泦浠诲姟鍒扮粺涓�璋冨害鍣�
        
        Args:
            source_id: 鏁版嵁婧怚D
            source_config: 鏁版嵁婧愰厤缃�
        """
        import asyncio
        
        try:
            # 绔炴�佹潯浠舵��鏌ュ凡绉诲埌涓婇潰鐨� should_collect 閫昏緫涓�
            
            # Setup event loop in this thread before accessing asyncio-based scheduler
            import asyncio
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                _loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_loop)

            scheduler = get_unified_scheduler()
            
            # 鍑嗗�囦换鍔℃暟鎹�
            task_data = {
                "source_id": source_id,
                "source_config": source_config,
                "collection_type": "scheduled",
                "submitted_at": datetime.now().isoformat()
            }
            
            # 瀹氫箟浠诲姟瀹屾垚鍥炶皟
            def on_task_completed(task_id: str, status: str, result: Any, error: str):
                """浠诲姟瀹屾垚鍥炶皟"""
                logger.info(f"馃幆 浠诲姟瀹屾垚鍥炶皟琚�璋冪敤: task_id={task_id}, source_id={source_id}, 鐘舵��={status}")
                
                # 銆愬叧閿�淇�澶嶃�戞洿鏂癟askManager涓�鐨勪换鍔＄姸鎬�
                try:
                    from src.core.orchestration.scheduler import get_unified_scheduler
                    from src.core.orchestration.scheduler.base import TaskStatus
                    tm = get_unified_scheduler()._task_manager
                    tm_status = TaskStatus.COMPLETED if status == "completed" else TaskStatus.FAILED
                    # 鐩存帴璁剧疆灞炴�э紙缁曡繃寮傛�ラ攣锛�
                    if task_id in tm._tasks:
                        tm._tasks[task_id].status = tm_status
                        tm._tasks[task_id].result = result
                        tm._tasks[task_id].completed_at = datetime.now()
                        logger.info(f"鉁� TaskManager浠诲姟鐘舵�佸凡鏇存柊: {task_id} -> {tm_status.name}")
                except Exception as tm_err:
                    logger.error(f"鉂� 鏇存柊TaskManager鐘舵�佸け璐�: {task_id}, error={tm_err}")
                
                if status == "completed":
                    logger.info(f"鉁� 鏁版嵁閲囬泦浠诲姟瀹屾垚: {source_id}, 缁撴灉={result}")
                    # 鏇存柊鏁版嵁婧愭渶鍚庨噰闆嗘椂闂�
                    try:
                        # 【修复】检查result是否包含新数据
                        result_has_data = False
                        if isinstance(result, dict):
                            records = result.get("records") or result.get("data") or result.get("collected_records", [])
                            count = result.get("collected_count") or result.get("count") or result.get("records_collected") or 0
                            result_has_data = (isinstance(records, (list, tuple)) and len(records) > 0) or (isinstance(count, int) and count > 0)
                        logger.info(f"   result类型: {type(result).__name__}, result_has_data: {result_has_data}")
                        self._update_source_collection_time(source_id, new_data_collected=result_has_data)
                        logger.info(f"鉁� 宸茶皟鐢╛update_source_collection_time: {source_id}")
                    except Exception as update_err:
                        logger.error(f"鉂� 璋冪敤_update_source_collection_time澶辫触: {source_id}, 閿欒��={update_err}", exc_info=True)
                    
                    # 鍙戝竷鏁版嵁閲囬泦瀹屾垚浜嬩欢锛岃Е鍙戝悗缁�涓氬姟娴佺▼
                    try:
                        from src.core.event_bus import get_event_bus
                        from src.core.event_bus.types import EventType
                        
                        event_bus = get_event_bus()
                        
                        # 鑾峰彇鏁版嵁婧愰厤缃�锛堢敤浜庣壒寰佸伐绋嬫ā鍧楋級
                        source_config_for_event = source_config.copy() if source_config else {}
                        
                        event_bus.publish(
                            EventType.DATA_COLLECTION_COMPLETED,
                            {
                                "source_id": source_id,
                                "task_id": task_id,
                                "status": status,
                                "result": result,
                                "source_config": source_config_for_event,  # 娣诲姞source_config浠ュ吋瀹圭壒寰佸伐绋嬫ā鍧�
                                "timestamp": datetime.now().isoformat(),
                                "collection_type": "scheduled"
                            },
                            source="data_collection_scheduler_manager"
                        )
                        logger.info(f"馃摙 鏁版嵁閲囬泦瀹屾垚浜嬩欢宸插彂甯�: {source_id}")
                    except Exception as event_err:
                        logger.warning(f"鈿狅笍 鍙戝竷鏁版嵁閲囬泦瀹屾垚浜嬩欢澶辫触锛堥潪鍏抽敭锛�: {source_id}, 閿欒��={event_err}")
                    
                elif status == "failed":
                    logger.error(f"鉂� 鏁版嵁閲囬泦浠诲姟澶辫触: {source_id}, 閿欒��={error}")
                    
                    # 鍙戝竷鏁版嵁閲囬泦澶辫触浜嬩欢
                    try:
                        from src.core.event_bus import get_event_bus
                        from src.core.event_bus.types import EventType
                        
                        event_bus = get_event_bus()
                        event_bus.publish(
                            EventType.DATA_COLLECTION_FAILED,
                            {
                                "source_id": source_id,
                                "task_id": task_id,
                                "status": status,
                                "error": error,
                                "timestamp": datetime.now().isoformat()
                            },
                            source="data_collection_scheduler_manager"
                        )
                        logger.info(f"馃摙 鏁版嵁閲囬泦澶辫触浜嬩欢宸插彂甯�: {source_id}")
                    except Exception as event_err:
                        logger.warning(f"鈿狅笍 鍙戝竷鏁版嵁閲囬泦澶辫触浜嬩欢澶辫触锛堥潪鍏抽敭锛�: {source_id}, 閿欒��={event_err}")
                
                # 淇�澶岯ug3: 浣跨敤淇濆瓨鐨勬彁浜ゆ椂闂存埑锛岀‘淇濅笌鎻愪氦鏃朵竴鑷�
                saved_time_key = self._completion_timestamps.get(f"{source_id}:{datetime.now().strftime('%Y%m%d')}", datetime.now().strftime('%Y%m%d'))
                task_key = f"{source_id}:{saved_time_key}"
                if task_key in self._submitted_tasks:
                    self._submitted_tasks.discard(task_key)
                    # 娓呯悊鏃堕棿鎴宠�板綍
                    self._completion_timestamps.pop(task_key, None)
                    logger.info(f"馃棏锔� 宸蹭粠鎻愪氦浠诲姟闆嗗悎绉婚櫎: {task_key}")
            
            # 鎻愪氦浠诲姟锛堝紓姝ユ柟娉曪級
            async def submit_task_async():
                # 鎻愪氦浠诲姟锛堜娇鐢ㄦ灇涓剧殑鍊硷紝鑰屼笉鏄�鏋氫妇鏈�韬�锛�
                task_id = await scheduler.submit_task(
                    task_type=TaskType.DATA_COLLECTION.value,
                    payload=task_data,
                    priority=TaskPriority.NORMAL
                )
                
                # 娉ㄥ唽浠诲姟瀹屾垚鍥炶皟
                worker_manager = scheduler._worker_manager
                worker_manager.register_task_callback(task_id, on_task_completed)
                
                return task_id
            
            # 鍦ㄥ悓姝ヤ笂涓嬫枃涓�杩愯�屽紓姝ヤ换鍔�
            task_id = None
            try:
                # 灏濊瘯鑾峰彇褰撳墠浜嬩欢寰�鐜�
                loop = asyncio.get_running_loop()
                # 濡傛灉宸茬粡鏈変簨浠跺惊鐜�锛屼娇鐢╮un_coroutine_threadsafe
                future = asyncio.run_coroutine_threadsafe(submit_task_async(), loop)
                task_id = future.result(timeout=30)
            except RuntimeError:
                # 娌℃湁浜嬩欢寰�鐜�锛屽垱寤烘柊鐨勪簨浠跺惊鐜�
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    task_id = loop.run_until_complete(submit_task_async())
                finally:
                    loop.close()
            
            if task_id:
                # 璁板綍宸叉彁浜ょ殑浠诲姟
                task_key = f"{source_id}:{datetime.now().strftime('%Y%m%d')}"
                self._submitted_tasks.add(task_key)
                
                self._stats["tasks_submitted"] += 1
                
                logger.info(f"鉁� 閲囬泦浠诲姟宸叉彁浜�: {task_id} (鏁版嵁婧�: {source_id})")
                
                # 楠岃瘉浠诲姟鏄�鍚﹁��璁板綍
                try:
                    stats = scheduler.get_statistics()
                    logger.info(f"馃搳 鎻愪氦鍚庤皟搴﹀櫒浠诲姟缁熻��: 鎬讳换鍔�={stats.get('total_tasks', 0)}, 寰呭�勭悊={stats.get('pending_tasks', 0)}")
                except Exception as stats_err:
                    logger.debug(f"鑾峰彇璋冨害鍣ㄧ粺璁″け璐�: {stats_err}")
            else:
                logger.error(f"鉂� 浠诲姟鎻愪氦澶辫触锛屾湭鑾峰彇鍒颁换鍔�ID: {source_id}")
            
        except Exception as e:
            logger.error(f"鉂� 鎻愪氦閲囬泦浠诲姟澶辫触 {source_id}: {e}", exc_info=True)
    
    def _update_source_collection_time(self, source_id: str, new_data_collected: bool = True):
        """
        更新数据源最后采集时间

        Args:
            source_id: 数据源ID
            new_data_collected: 是否有新数据被采集。为True时才更新last_collection，
             False时只更新last_test（表示健康检查时间）。
        """
        try:
            from src.gateway.web.data_source_config_manager import get_data_source_config_manager
            
            logger.info(f"馃攧 寮�濮嬫洿鏂版暟鎹�婧愭渶鍚庨噰闆嗘椂闂�: {source_id}")
            
            config_manager = get_data_source_config_manager()
            
            # 鏇存柊鏁版嵁婧愰厤缃�
            source_config = config_manager.get_data_source(source_id)
            if source_config:
                now = datetime.now()
                now_iso = now.isoformat()
                now_str = now.strftime('%Y-%m-%d %H:%M:%S')
                
                old_collection_time = source_config.get("last_collection_time")
                old_test_time = source_config.get("last_test")
                
                # 【修复】只在有新数据时才更新last_collection，避免采集失败仍更新时间戳
                source_config["last_test"] = now_str
                if new_data_collected:
                    source_config["last_collection_time"] = now_iso
                    source_config["last_collection"] = now_iso
                    logger.info(f"   last_collection 已更新（有新数据）")
                else:
                    logger.info(f"   last_collection 保持不变（本次无新数据）")

                logger.info(f"准备保存配置 - 数据源: {source_id}, new_data_collected={new_data_collected}")
                logger.info(f"   last_test: {old_test_time} -> {now_str}")
                
                success = config_manager.update_data_source(source_id, source_config)
                
                if success:
                    logger.info(f"鉁� 鏇存柊鏁版嵁婧愭椂闂存垚鍔�: {source_id}")
                    # 銆愬叧閿�淇�澶嶃�戦噸鏂板姞杞介厤缃�浠ュ悓姝ュ唴瀛樼紦瀛橈紝纭�淇濅笅娆�get_data_source()璇诲埌鏈�鏂板��
                    try:
                        config_manager.reload_config()
                        reloaded = config_manager.get_data_source(source_id)
                        if reloaded:
                            logger.info("reload楠岃瘉: %s last_test=%s" % (source_id, reloaded.get('last_test')))
                        else:
                            logger.warning("閲嶆柊鍔犺浇鍚庢暟鎹�婧愪笉瀛樺湪: %s" % source_id)
                    except Exception as reload_err:
                        logger.warning("閰嶇疆閲嶆柊鍔犺浇澶辫触: %s" % reload_err)
                else:
                    logger.error(f"鉂� 鏇存柊鏁版嵁婧愭椂闂村け璐�: {source_id} - update_data_source杩斿洖False")
            else:
                logger.warning(f"鈿狅笍 鏈�鎵惧埌鏁版嵁婧愰厤缃�: {source_id}")
            
        except Exception as e:
            logger.error(f"鉂� 鏇存柊鏁版嵁婧愰噰闆嗘椂闂村け璐� {source_id}: {e}", exc_info=True)
    
    def _cleanup_completed_tasks(self):
        """娓呯悊宸插畬鎴愮殑浠诲姟璁板綍"""
        # 鍙�淇濈暀鏈�杩�3澶╃殑浠诲姟璁板綍
        current_date = datetime.now().strftime('%Y%m%d')
        tasks_to_remove = []
        
        for task_key in self._submitted_tasks:
            # task_key 鏍煎紡: source_id:YYYYMMDD
            if ':' in task_key:
                date_part = task_key.split(':')[-1]
                if date_part != current_date:
                    tasks_to_remove.append(task_key)
        
        for task_key in tasks_to_remove:
            self._submitted_tasks.discard(task_key)
        
        if tasks_to_remove:
            logger.debug(f"娓呯悊浜� {len(tasks_to_remove)} 涓�鍘嗗彶浠诲姟璁板綍")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        鑾峰彇璋冨害绠＄悊鍣ㄧ粺璁′俊鎭�
        
        Returns:
            Dict: 缁熻�′俊鎭�
        """
        return {
            "running": self._running,
            "check_interval": self._check_interval,
            "total_checks": self._stats["total_checks"],
            "tasks_submitted": self._stats["tasks_submitted"],
            "sources_checked": self._stats["sources_checked"],
            "last_check_time": self._stats["last_check_time"],
            "next_check_time": self._stats["next_check_time"],
            "pending_tasks_count": len(self._submitted_tasks)
        }
    
    def force_check(self):
        """
        寮哄埗绔嬪嵆鎵ц�屼竴娆℃��鏌�
        """
        logger.info("馃殌 寮哄埗鎵ц�屾暟鎹�婧愭��鏌�")
        self._check_and_schedule()


# 鍏ㄥ眬瀹炰緥
_scheduler_manager: Optional[DataCollectionSchedulerManager] = None


def get_scheduler_manager() -> DataCollectionSchedulerManager:
    """
    鑾峰彇鍏ㄥ眬璋冨害绠＄悊鍣ㄥ疄渚嬶紙鍗曚緥妯″紡锛岃嚜鍔ㄥ惎鍔ㄨ皟搴﹀櫒锛�
    """
    global _scheduler_manager
    if _scheduler_manager is None:
        _scheduler_manager = DataCollectionSchedulerManager()
        _scheduler_manager.start()  # 鑷�鍔ㄥ惎鍔ㄨ皟搴︾嚎绋�
    return _scheduler_manager


def start_auto_collection() -> bool:
    """
    鍚�鍔ㄨ嚜鍔ㄩ噰闆�
    
    Returns:
        bool: 鏄�鍚︽垚鍔熷惎鍔�
    """
    manager = get_scheduler_manager()
    return manager.start()


def stop_auto_collection() -> bool:
    """
    鍋滄�㈣嚜鍔ㄩ噰闆�
    
    Returns:
        bool: 鏄�鍚︽垚鍔熷仠姝�
    """
    manager = get_scheduler_manager()
    return manager.stop()


def get_auto_collection_status() -> Dict[str, Any]:
    """
    鑾峰彇鑷�鍔ㄩ噰闆嗙姸鎬�
    
    Returns:
        Dict: 鐘舵�佷俊鎭�
    """
    manager = get_scheduler_manager()
    return manager.get_stats()


# 娴嬭瘯浠ｇ爜
if __name__ == "__main__":
    print("=" * 60)
    print("鏁版嵁閲囬泦璋冨害绠＄悊鍣ㄦ祴璇�")
    print("=" * 60)
    
    # 鑾峰彇绠＄悊鍣ㄥ疄渚�
    manager = get_scheduler_manager()
    
    # 娴嬭瘯鍚�鍔�
    print("\n1. 娴嬭瘯鍚�鍔ㄨ皟搴︾�＄悊鍣�")
    result = manager.start()
    print(f"鉁� 鍚�鍔ㄧ粨鏋�: {result}")
    
    # 鑾峰彇鐘舵��
    print("\n2. 鑾峰彇鐘舵��")
    stats = manager.get_stats()
    print(f"杩愯�岀姸鎬�: {stats['running']}")
    print(f"妫�鏌ラ棿闅�: {stats['check_interval']}绉�")
    
    # 绛夊緟鍑犵��
    print("\n3. 绛夊緟5绉�...")
    time.sleep(5)
    
    # 鍐嶆�¤幏鍙栫姸鎬�
    stats = manager.get_stats()
    print(f"妫�鏌ユ�℃暟: {stats['total_checks']}")
    print(f"鏈�鍚庢��鏌�: {stats['last_check_time']}")
    
    # 娴嬭瘯鍋滄��
    print("\n4. 娴嬭瘯鍋滄�㈣皟搴︾�＄悊鍣�")
    result = manager.stop()
    print(f"鉁� 鍋滄�㈢粨鏋�: {result}")
    
    print("\n" + "=" * 60)
    print("娴嬭瘯瀹屾垚")
    print("=" * 60)

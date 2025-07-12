import os
import sys
import re
import traceback

def main():
    file_path = "src/infrastructure/config/config_watcher.py"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 - {file_path}")
        return 1

    try:
        # 读取当前文件内容
        print(f"正在读取文件：{file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"成功读取文件")

        # 修改1: 增强_process_pending_events方法
        process_pending_pattern = re.compile(r'def _process_pending_events\(self\):.*?if self._pending_events:', re.DOTALL)
        if process_pending_pattern.search(content):
            print("找到_process_pending_events方法，进行增强")

            # 创建增强版的_process_pending_events方法
            new_process_pending = '''            def _process_pending_events(self):
                """处理所有待处理的事件"""
                try:
                    logger.info(f"开始处理待处理事件，数量: {len(self._pending_events)}")
                    current_time = time.time()
                    
                    # 处理每个待处理事件
                    for file_path, event_time in list(self._pending_events.items()):
                        # 跳过太新的事件（可能还在写入）
                        if current_time - event_time < self.outer._debounce_interval * 0.2:  # 进一步缩短等待时间
                            logger.debug(f"跳过太新的事件: {file_path}, 间隔: {current_time - event_time:.3f}秒")
                            continue
                            
                        # 确保文件已完全写入
                        try:
                            # 尝试打开文件以确保写入完成
                            with open(file_path, 'r') as f:
                                content = f.read()
                                logger.info(f"变更后文件内容长度: {len(content)} 字节")
                                
                                # 直接处理配置变更
                                logger.info(f"准备直接处理配置文件变更: {file_path}")
                                
                                # 获取环境名
                                env = os.path.splitext(os.path.basename(file_path))[0]
                                
                                # 验证文件内容是否有效
                                try:
                                    import json
                                    config_data = json.loads(content)
                                    logger.info(f"成功解析JSON配置: {file_path}")
                                    
                                    # 直接更新配置缓存
                                    if hasattr(self.outer, '_manager') and self.outer._manager:
                                        # 清除配置缓存
                                        if env in self.outer._manager._configs:
                                            del self.outer._manager._configs[env]
                                            logger.info(f"已清除环境 {env} 的配置缓存")
                                        
                                        # 重新加载配置
                                        try:
                                            new_config = self.outer._manager._load_config(env)
                                            logger.info(f"配置重新加载完成: {env}")
                                            
                                            # 触发回调
                                            if env in self.outer._watchers:
                                                for key, callback in self.outer._watchers[env].items():
                                                    try:
                                                        # 获取最新值
                                                        value = self.outer._manager.get(key, env)
                                                        logger.info(f"触发回调: {env}.{key} = {value}")
                                                        callback(value)
                                                        logger.info(f"回调成功: {env}.{key}")
                                                    except Exception as e:
                                                        logger.error(f"触发回调失败: {env}.{key}, 错误: {str(e)}", exc_info=True)
                                        except Exception as e:
                                            logger.error(f"重新加载配置失败: {str(e)}", exc_info=True)
                                    
                                    # 如果有全局回调，也触发它
                                    if self.outer.on_config_changed:
                                        try:
                                            self.outer.on_config_changed(file_path)
                                            logger.info(f"全局配置变更回调触发成功: {file_path}")
                                        except Exception as e:
                                            logger.error(f"全局配置变更回调失败: {str(e)}", exc_info=True)
                                    
                                except ValueError as e:
                                    logger.warning(f"JSON解析失败，尝试YAML格式: {str(e)}")
                                    # 尝试YAML格式
                                    try:
                                        import yaml
                                        config_data = yaml.safe_load(content)
                                        logger.info(f"成功解析YAML配置: {file_path}")
                                        
                                        # 同样的处理逻辑...
                                        if hasattr(self.outer, '_manager') and self.outer._manager:
                                            if env in self.outer._manager._configs:
                                                del self.outer._manager._configs[env]
                                            new_config = self.outer._manager._load_config(env)
                                            # 触发回调...
                                    except Exception as yaml_e:
                                        logger.error(f"YAML解析也失败: {str(yaml_e)}", exc_info=True)
                                        # 尝试使用原始方法处理
                                        self.outer._handle_config_change(file_path)
                                
                                # 从待处理列表中移除
                                del self._pending_events[file_path]
                                logger.info(f"配置文件变更处理完成: {file_path}")
                        except Exception as e:
                            logger.error(f"处理待处理事件失败: {str(e)}", exc_info=True)
                            # 如果是权限或IO错误，可能文件仍在写入，保留在待处理列表
                            if isinstance(e, (PermissionError, IOError)):
                                logger.warning(f"文件可能仍在写入，保留在待处理列表: {file_path}")
                            else:
                                # 其他错误，从列表中移除
                                del self._pending_events[file_path]
                    
                    # 如果还有待处理事件，设置新的定时器
                    if self._pending_events:'''

            content = process_pending_pattern.sub(new_process_pending, content)
            print("_process_pending_events方法增强完成")
        else:
            print("未找到_process_pending_events方法，跳过增强")

        # 修改2: 增强on_modified方法中的Windows平台处理
        on_modified_pattern = re.compile(r'def on_modified\(self, event\):.*?# Windows平台增强处理.*?if os\.name == \'nt\':', re.DOTALL)
        if on_modified_pattern.search(content):
            print("找到on_modified方法中的Windows平台处理，进行增强")

            # 创建增强版的Windows平台处理
            new_windows_handling = '''            def on_modified(self, event):
                try:
                    if event.is_directory:
                        return
                        
                    # 使用绝对路径确保一致性
                    abs_path = os.path.abspath(event.src_path)
                    filename = os.path.basename(abs_path)
                    logger.debug(f"检测到文件变更事件: {filename} (env: {self.env})")
                    
                    if filename in [f"{self.env}.json", f"{self.env}.yaml"]:
                        current_time = time.time()
                        
                        # 记录事件
                        self._pending_events[abs_path] = current_time
                        logger.info(f"记录配置文件变更事件: {abs_path}, 时间: {current_time}")
                        
                        # 取消之前的定时器
                        if self._event_timer:
                            self._event_timer.cancel()
                            
                        # 设置新的定时器，延迟处理所有待处理事件
                        self._event_timer = threading.Timer(
                            self.outer._debounce_interval * 0.2,  # 进一步缩短延迟时间
                            self._process_pending_events
                        )
                        self._event_timer.daemon = True
                        self._event_timer.name = "ConfigFileHandlerTimer"
                        self._event_timer.start()
                        
                        # Windows平台增强处理
                        if os.name == 'nt':'''

            content = on_modified_pattern.sub(new_windows_handling, content)
            print("on_modified方法中的Windows平台处理增强完成")
        else:
            print("未找到on_modified方法中的Windows平台处理，跳过增强")

        # 写回文件
        print(f"正在写入文件：{file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("成功写入文件")
        return 0

    except Exception as e:
        print(f"错误：{str(e)}")
        print("详细错误信息：")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

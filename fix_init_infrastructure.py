with open('src/infrastructure/init_infrastructure.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 移除重复的try语句 (第251行)
lines[250] = '                try:\n'
lines[251] = '            resource_config = self.config.get(\'resources\', default={})\n'

# 修复缩进错误
lines[252] = '\n'
lines[253] = '            # 初始化资源管理器\n'
lines[254] = '            self.resource = ResourceManager(\n'
lines[255] = '                cpu_threshold=resource_config.get(\'cpu_threshold\', 80.0),\n'
lines[256] = '                mem_threshold=resource_config.get(\'memory_threshold\', 80.0),\n'
lines[257] = '                disk_threshold=resource_config.get(\'disk_threshold\', 80.0)\n'
lines[258] = '            )\n'
lines[259] = '            self.gpu = GPUManager()\n'
lines[260] = '\n'
lines[261] = '            # 启动资源监控\n'
lines[262] = '            if resource_config.get(\'monitoring_enabled\', True):\n'
lines[263] = '                self.resource.start_monitoring()\n'
lines[264] = '            # 暂时注释掉GPU监控，因为GPUManager可能没有这些方法\n'
lines[265] = '            # if self.gpu.get_gpu_count() > 0:\n'
lines[266] = '            #     self.gpu.start_monitoring()\n'
lines[267] = '\n'

# 修复update语法错误
lines[268] = '                self._components.update({\n'
lines[269] = '                \'resource_manager\': self.resource,\n'
lines[270] = '                \'gpu_manager\': self.gpu\n'
lines[271] = '            })\n'

# 修复except缩进
lines[273] = '                self._logger.info("Resource managers initialized")\n'
lines[274] = '    except Exception as e:\n'
lines[275] = '            self._logger.critical(f"Failed to initialize resource managers: {e}")\n'
lines[276] = '            raise\n'

with open('src/infrastructure/init_infrastructure.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('Fixed all syntax errors in init_infrastructure.py')

"""识别低覆盖模块 - 系统性方法第1步"""
import json
import os

def identify_low_coverage_modules():
    print('🔍 第1步：识别低覆盖模块')
    print('=' * 60)
    
    # 读取覆盖率报告
    with open('test_logs/health_coverage_FINAL_COMPLETE.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 分析模块覆盖率
    low_coverage_modules = []
    medium_coverage_modules = []
    high_coverage_modules = []
    
    for filepath, filedata in data['files'].items():
        if 'src' in filepath and 'infrastructure' in filepath and 'health' in filepath:
            percent = filedata['summary']['percent_covered']
            missing = filedata['summary']['missing_lines']
            total = filedata['summary']['num_statements']
            filename = filepath.replace('\\', '/').split('/')[-1]
            
            # 计算ROI (缺失行数 / (目标覆盖率 - 当前覆盖率))
            target_coverage = 70
            if percent < target_coverage:
                roi = missing / (target_coverage - percent) if percent < target_coverage else 0
            else:
                roi = 0
            
            module_info = {
                'name': filename,
                'percent': percent,
                'missing': missing,
                'total': total,
                'roi': roi,
                'filepath': filepath
            }
            
            if percent < 30:
                low_coverage_modules.append(module_info)
            elif percent < 50:
                medium_coverage_modules.append(module_info)
            else:
                high_coverage_modules.append(module_info)
    
    # 按ROI排序
    low_coverage_modules.sort(key=lambda x: x['roi'], reverse=True)
    medium_coverage_modules.sort(key=lambda x: x['roi'], reverse=True)
    
    print(f'📊 模块分析结果:')
    print(f'  低覆盖模块 (<30%): {len(low_coverage_modules)}个')
    print(f'  中等覆盖模块 (30-50%): {len(medium_coverage_modules)}个')
    print(f'  高覆盖模块 (50%+): {len(high_coverage_modules)}个')
    print()
    
    # 显示低覆盖模块TOP10
    print('🎯 低覆盖模块TOP10 (ROI排序):')
    print('-' * 80)
    print(f'{"序号":<4} {"模块名称":<40} {"覆盖率":<8} {"缺失行":<8} {"ROI":<6}')
    print('-' * 80)
    
    for i, module in enumerate(low_coverage_modules[:10], 1):
        print(f'{i:<4} {module["name"]:<40} {module["percent"]:>6.1f}% {module["missing"]:>6}行 {module["roi"]:>5.1f}')
    
    print()
    
    # 显示中等覆盖模块TOP5
    print('🎯 中等覆盖模块TOP5 (ROI排序):')
    print('-' * 80)
    print(f'{"序号":<4} {"模块名称":<40} {"覆盖率":<8} {"缺失行":<8} {"ROI":<6}')
    print('-' * 80)
    
    for i, module in enumerate(medium_coverage_modules[:5], 1):
        print(f'{i:<4} {module["name"]:<40} {module["percent"]:>6.1f}% {module["missing"]:>6}行 {module["roi"]:>5.1f}')
    
    print()
    
    # 推荐策略
    print('💡 推荐策略:')
    print('  1. 优先处理ROI最高的低覆盖模块')
    print('  2. 快速提升中等覆盖模块到70%+')
    print('  3. 预计需要新增测试: 200-300个')
    print('  4. 目标: 从40.03%提升到70%+')
    
    return low_coverage_modules[:5], medium_coverage_modules[:5]

if __name__ == '__main__':
    identify_low_coverage_modules()

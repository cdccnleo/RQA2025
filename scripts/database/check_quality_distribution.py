"""
检查特征质量分布仪表盘数据
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def check_quality_data():
    """检查数据库中的质量数据"""
    print("\n" + "="*80)
    print("步骤1: 检查数据库中的质量数据")
    print("="*80)
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. 检查 quality_score 字段是否存在
        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'feature_store'
            AND column_name = 'quality_score'
        """)
        
        if cursor.fetchone():
            print("\n✅ feature_store 表包含 quality_score 字段")
        else:
            print("\n⚠️  feature_store 表不包含 quality_score 字段")
            return None
        
        # 2. 查询质量评分数据
        cursor.execute("""
            SELECT quality_score, COUNT(*) as count
            FROM feature_store
            WHERE quality_score IS NOT NULL
            GROUP BY quality_score
            ORDER BY quality_score
        """)
        
        quality_data = cursor.fetchall()
        
        if quality_data:
            print(f"\n📊 质量评分分布（按具体分值）:")
            print(f"{'质量评分':<15} {'特征数量':<10}")
            print("-" * 30)
            for score, count in quality_data:
                print(f"{score:<15.2f} {count:<10}")
        else:
            print("\n⚠️  未发现质量评分数据")
        
        # 3. 按区间统计质量分布
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN quality_score >= 0.9 THEN '优秀 (0.9-1.0)'
                    WHEN quality_score >= 0.8 THEN '良好 (0.8-0.9)'
                    WHEN quality_score >= 0.7 THEN '一般 (0.7-0.8)'
                    WHEN quality_score >= 0.6 THEN '较差 (0.6-0.7)'
                    ELSE '差 (<0.6)'
                END as quality_range,
                COUNT(*) as count,
                AVG(quality_score) as avg_score
            FROM feature_store
            WHERE quality_score IS NOT NULL
            GROUP BY 
                CASE 
                    WHEN quality_score >= 0.9 THEN '优秀 (0.9-1.0)'
                    WHEN quality_score >= 0.8 THEN '良好 (0.8-0.9)'
                    WHEN quality_score >= 0.7 THEN '一般 (0.7-0.8)'
                    WHEN quality_score >= 0.6 THEN '较差 (0.6-0.7)'
                    ELSE '差 (<0.6)'
                END
            ORDER BY avg_score DESC
        """)
        
        distribution = cursor.fetchall()
        
        if distribution:
            print(f"\n📊 质量评分分布（按区间）:")
            print(f"{'质量区间':<25} {'特征数量':<10} {'平均评分':<10}")
            print("-" * 50)
            for range_name, count, avg_score in distribution:
                print(f"{range_name:<25} {count:<10} {avg_score:<10.2f}")
        
        # 4. 统计缺失质量评分的特征
        cursor.execute("""
            SELECT COUNT(*)
            FROM feature_store
            WHERE quality_score IS NULL
        """)
        
        null_count = cursor.fetchone()[0]
        print(f"\n📊 缺失质量评分的特征数: {null_count}")
        
        # 5. 总体统计
        cursor.execute("""
            SELECT 
                COUNT(*) as total_count,
                COUNT(quality_score) as scored_count,
                AVG(quality_score) as avg_score,
                MIN(quality_score) as min_score,
                MAX(quality_score) as max_score
            FROM feature_store
        """)
        
        total, scored, avg, min_s, max_s = cursor.fetchone()
        print(f"\n📊 总体统计:")
        print(f"   - 总特征数: {total}")
        print(f"   - 有评分特征数: {scored}")
        print(f"   - 平均评分: {avg:.2f}" if avg else "   - 平均评分: N/A")
        print(f"   - 最低评分: {min_s:.2f}" if min_s else "   - 最低评分: N/A")
        print(f"   - 最高评分: {max_s:.2f}" if max_s else "   - 最高评分: N/A")
        
        cursor.close()
        return distribution
        
    except Exception as e:
        print(f"\n❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if conn:
            return_db_connection(conn)


def check_frontend_api():
    """检查前端API"""
    print("\n" + "="*80)
    print("步骤2: 检查前端API")
    print("="*80)
    
    try:
        # 尝试调用API获取质量分布数据
        import requests
        
        # 检查API端点
        api_url = "http://localhost:8000/api/v1/features/engineering/quality/distribution"
        print(f"\n📋 尝试调用API: {api_url}")
        
        response = requests.get(api_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ API调用成功")
            print(f"\n📊 API返回数据:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return data
        else:
            print(f"\n⚠️  API返回错误: {response.status_code}")
            print(f"   响应内容: {response.text}")
            return None
            
    except Exception as e:
        print(f"\n⚠️  API调用失败: {e}")
        print("   这是正常的，如果服务没有运行")
        return None


def check_frontend_logic():
    """检查前端展示逻辑"""
    print("\n" + "="*80)
    print("步骤3: 检查前端展示逻辑")
    print("="*80)
    
    print("\n📋 前端展示逻辑分析:")
    print("   文件: web-static/feature-engineering-monitor.html")
    print("   需要检查的函数:")
    print("      - renderQualityDistribution() - 渲染质量分布图表")
    print("      - loadQualityData() - 加载质量数据")
    
    print("\n📋 预期行为:")
    print("   1. 从API获取质量分布数据")
    print("   2. 按区间统计特征数量")
    print("   3. 使用Chart.js渲染柱状图或饼图")
    print("   4. 不同质量区间使用不同颜色标识")
    
    print("\n📋 质量区间划分标准:")
    print("   - 优秀: 0.9-1.0 (绿色)")
    print("   - 良好: 0.8-0.9 (浅绿色)")
    print("   - 一般: 0.7-0.8 (黄色)")
    print("   - 较差: 0.6-0.7 (橙色)")
    print("   - 差: <0.6 (红色)")


def compare_data(db_data, api_data):
    """对比数据库数据与API数据"""
    print("\n" + "="*80)
    print("步骤4: 数据一致性对比")
    print("="*80)
    
    if db_data and api_data:
        print("\n📊 对比数据库与API数据:")
        # 这里可以添加具体的对比逻辑
        print("   ✅ 数据对比功能待实现")
    elif db_data:
        print("\n📊 数据库数据:")
        for range_name, count, avg_score in db_data:
            print(f"   {range_name}: {count} 个特征")
    else:
        print("\n⚠️  无法对比，缺少数据")


def main():
    print("🚀 开始检查特征质量分布仪表盘")
    
    # 步骤1: 检查数据库
    db_data = check_quality_data()
    
    # 步骤2: 检查前端API
    api_data = check_frontend_api()
    
    # 步骤3: 检查前端逻辑
    check_frontend_logic()
    
    # 步骤4: 对比数据
    compare_data(db_data, api_data)
    
    print("\n" + "="*80)
    print("✅ 检查完成")
    print("="*80)


if __name__ == "__main__":
    main()

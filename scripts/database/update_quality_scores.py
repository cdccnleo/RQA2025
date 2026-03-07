"""
更新现有特征的质量评分为差异化评分
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection
from src.features.quality import get_feature_quality_score, get_quality_category


def update_existing_quality_scores():
    """更新现有特征的质量评分"""
    print("\n" + "="*80)
    print("📝 更新现有特征的质量评分")
    print("="*80)

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 获取所有特征
        cursor.execute("SELECT feature_id, feature_name, quality_score FROM feature_store")
        features = cursor.fetchall()

        print(f"\n📊 共找到 {len(features)} 个特征")

        # 统计更新前的分布
        cursor.execute("""
            SELECT quality_score, COUNT(*) as count
            FROM feature_store
            GROUP BY quality_score
            ORDER BY quality_score
        """)
        before_distribution = cursor.fetchall()

        print("\n📊 更新前的质量评分分布:")
        for score, count in before_distribution:
            print(f"   - {score}: {count} 个特征")

        # 更新每个特征的质量评分
        updated_count = 0
        category_counts = {}

        for feature_id, feature_name, old_score in features:
            # 计算新的质量评分
            new_score = get_feature_quality_score(feature_name)
            category = get_quality_category(new_score)
            category_counts[category] = category_counts.get(category, 0) + 1

            # 更新数据库
            cursor.execute(
                "UPDATE feature_store SET quality_score = %s WHERE feature_id = %s",
                (new_score, feature_id)
            )
            updated_count += 1

        conn.commit()

        print(f"\n✅ 已更新 {updated_count} 个特征的质量评分")

        # 统计更新后的分布
        cursor.execute("""
            SELECT
                CASE
                    WHEN quality_score >= 0.9 THEN '优秀 (0.9+)'
                    WHEN quality_score >= 0.8 THEN '良好 (0.8-0.9)'
                    WHEN quality_score >= 0.7 THEN '一般 (0.7-0.8)'
                    ELSE '较差 (<0.7)'
                END as quality_range,
                COUNT(*) as count,
                AVG(quality_score) as avg_score
            FROM feature_store
            GROUP BY quality_range
            ORDER BY avg_score DESC
        """)
        after_distribution = cursor.fetchall()

        print("\n📊 更新后的质量评分分布:")
        print(f"{'质量区间':<25} {'特征数量':<10} {'平均评分':<10}")
        print("-" * 50)
        for range_name, count, avg_score in after_distribution:
            print(f"{range_name:<25} {count:<10} {avg_score:<10.2f}")

        # 显示各质量等级的统计
        print("\n📊 各质量等级统计:")
        for category, count in sorted(category_counts.items()):
            print(f"   - {category}: {count} 个特征")

        cursor.close()

        print("\n" + "="*80)
        print("✅ 质量评分更新完成")
        print("="*80)

        return updated_count

    except Exception as e:
        print(f"\n❌ 更新失败: {e}")
        import traceback
        traceback.print_exc()
        if conn:
            conn.rollback()
        return 0
    finally:
        if conn:
            return_db_connection(conn)


def main():
    print("🚀 开始更新特征质量评分")
    update_existing_quality_scores()


if __name__ == "__main__":
    main()

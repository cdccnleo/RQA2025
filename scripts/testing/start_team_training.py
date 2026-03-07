#!/usr/bin/env python3
"""
RQA2025 团队培训启动脚本
帮助开始培训活动
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path


class TeamTrainingStarter:
    """团队培训启动器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.training_dir = self.project_root / "docs" / "training"

    def check_training_materials(self):
        """检查培训材料是否完整"""
        print("🔍 检查培训材料...")

        required_files = [
            "team_training_plan.md",
            "quick_start_guide.md",
            "training_implementation_guide.md"
        ]

        missing_files = []
        for file in required_files:
            file_path = self.training_dir / file
            if not file_path.exists():
                missing_files.append(file)
            else:
                print(f"✅ {file}")

        if missing_files:
            print(f"❌ 缺少培训材料: {', '.join(missing_files)}")
            return False

        print("✅ 所有培训材料已准备就绪")
        return True

    def generate_training_announcement(self):
        """生成培训通知"""
        announcement = f"""
# 📢 RQA2025 团队培训启动通知

## 📅 培训时间
**开始日期**: {datetime.now().strftime('%Y-%m-%d')}
**培训周期**: 3周
**培训地点**: 会议室A

## 📋 培训安排

### 第一周：基础培训
- **Day 1** ({datetime.now().strftime('%Y-%m-%d')}): 自动化工具使用培训 (2小时)
- **Day 2** ({(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}): 预提交钩子使用说明 (1小时)
- **Day 3** ({(datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')}): 覆盖率仪表板查看方法 (1小时)

### 第二周：进阶培训
- **Day 1** ({(datetime.now() + timedelta(days=8)).strftime('%Y-%m-%d')}): CI/CD流水线使用指南 (1.5小时)
- **Day 2** ({(datetime.now() + timedelta(days=9)).strftime('%Y-%m-%d')}): 质量门禁使用说明 (1小时)
- **Day 3** ({(datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')}): 综合练习和答疑 (2小时)

### 第三周：实战演练
- **Day 1-3** ({(datetime.now() + timedelta(days=15)).strftime('%Y-%m-%d')} 至 {(datetime.now() + timedelta(days=17)).strftime('%Y-%m-%d')}): 实际项目中的工具使用 (每天2小时)
- **Day 4-5** ({(datetime.now() + timedelta(days=18)).strftime('%Y-%m-%d')} 至 {(datetime.now() + timedelta(days=19)).strftime('%Y-%m-%d')}): 问题解决和优化 (每天1小时)

## 📚 培训材料
- **团队培训计划**: `docs/training/team_training_plan.md`
- **快速使用指南**: `docs/training/quick_start_guide.md`
- **培训实施指南**: `docs/training/training_implementation_guide.md`

## 🎯 培训目标
1. 掌握自动化测试工具的使用方法
2. 学会使用预提交钩子进行代码质量检查
3. 学会查看和解读覆盖率仪表板
4. 了解CI/CD流水线的工作原理和使用方法
5. 理解质量门禁的重要性和使用方法

## 📊 预期成果
- 所有团队成员熟练掌握工具使用
- 自动化流程100%运行
- 覆盖率监控机制建立
- 团队自动化文化建立

## 🔄 培训流程
1. **课前准备**: 阅读培训材料
2. **课堂学习**: 理论讲解 + 实操演练
3. **课后练习**: 独立练习和团队协作
4. **效果评估**: 定期评估和反馈

## 📝 注意事项
- 请提前阅读相关培训材料
- 培训期间请积极参与互动
- 如有问题请及时提出
- 培训结束后请填写反馈表

## 📞 联系方式
- **培训负责人**: 技术负责人
- **技术支持**: 资深开发工程师
- **问题反馈**: 请通过邮件或即时通讯工具联系

---
**让我们共同努力，建立高质量的自动化测试文化！**
"""
        return announcement

    def generate_participant_list(self):
        """生成参与者名单模板"""
        participants = """
# 👥 RQA2025 团队培训参与者名单

## 📋 参与者信息

| 姓名 | 部门 | 角色 | 联系方式 | 培训状态 |
|------|------|------|----------|----------|
| 张三 | 开发部 | 高级开发工程师 | zhangsan@company.com | 待确认 |
| 李四 | 开发部 | 开发工程师 | lisi@company.com | 待确认 |
| 王五 | 测试部 | 测试工程师 | wangwu@company.com | 待确认 |
| 赵六 | 开发部 | 初级开发工程师 | zhaoliu@company.com | 待确认 |
| 孙七 | 测试部 | 自动化测试工程师 | sunqi@company.com | 待确认 |

## 📊 培训统计
- **总人数**: 5人
- **开发人员**: 3人
- **测试人员**: 2人
- **已确认**: 0人
- **待确认**: 5人

## 📅 培训时间安排
- **第一周**: 基础培训 (4小时)
- **第二周**: 进阶培训 (4.5小时)
- **第三周**: 实战演练 (10小时)
- **总计**: 18.5小时

## 🎯 培训目标
- 所有参与者熟练掌握自动化工具
- 建立团队自动化文化
- 提升项目整体质量

---
**请参与者及时确认参加培训！**
"""
        return participants

    def generate_training_checklist(self):
        """生成培训检查清单"""
        checklist = """
# ✅ RQA2025 团队培训检查清单

## 📋 培训前准备

### 1. 培训材料准备
- [ ] 团队培训计划已制定
- [ ] 快速使用指南已编写
- [ ] 培训实施指南已准备
- [ ] 所有培训材料已分发

### 2. 培训环境准备
- [ ] 培训场地已预订
- [ ] 投影设备已测试
- [ ] 网络连接已确认
- [ ] 培训环境已搭建

### 3. 参与者确认
- [ ] 参与者名单已确定
- [ ] 培训时间已通知
- [ ] 参与者已确认参加
- [ ] 培训分组已安排

### 4. 讲师准备
- [ ] 讲师已确定
- [ ] 培训内容已熟悉
- [ ] 演示环境已准备
- [ ] 答疑材料已准备

## 📅 培训执行

### 1. 第一周：基础培训
- [ ] Day 1: 自动化工具使用培训
- [ ] Day 2: 预提交钩子使用说明
- [ ] Day 3: 覆盖率仪表板查看方法

### 2. 第二周：进阶培训
- [ ] Day 1: CI/CD流水线使用指南
- [ ] Day 2: 质量门禁使用说明
- [ ] Day 3: 综合练习和答疑

### 3. 第三周：实战演练
- [ ] Day 1-3: 实际项目中的工具使用
- [ ] Day 4-5: 问题解决和优化

## 📊 培训评估

### 1. 学习效果评估
- [ ] 知识掌握程度评估
- [ ] 技能应用能力评估
- [ ] 工具使用熟练度评估
- [ ] 问题解决能力评估

### 2. 培训反馈收集
- [ ] 培训内容评估
- [ ] 培训方式评估
- [ ] 培训效果评估
- [ ] 改进建议收集

### 3. 效果跟踪
- [ ] 工具使用频率统计
- [ ] 覆盖率变化趋势
- [ ] 质量指标改善情况
- [ ] 团队协作效果

## 🎯 培训后跟进

### 1. 持续支持
- [ ] 建立技术支持渠道
- [ ] 定期答疑和指导
- [ ] 分享最佳实践
- [ ] 解决使用问题

### 2. 效果评估
- [ ] 月度效果评估
- [ ] 季度总结报告
- [ ] 年度效果分析
- [ ] 持续改进计划

### 3. 文化建立
- [ ] 建立自动化文化
- [ ] 推广最佳实践
- [ ] 鼓励创新改进
- [ ] 建立激励机制

---
**培训成功的关键是持续跟进和效果评估！**
"""
        return checklist

    def save_training_files(self):
        """保存培训启动文件"""
        # 创建培训启动目录
        training_start_dir = self.project_root / "docs" / "training" / "startup"
        training_start_dir.mkdir(parents=True, exist_ok=True)

        # 保存培训通知
        announcement = self.generate_training_announcement()
        announcement_file = training_start_dir / "training_announcement.md"
        with open(announcement_file, 'w', encoding='utf-8') as f:
            f.write(announcement)

        # 保存参与者名单
        participants = self.generate_participant_list()
        participants_file = training_start_dir / "participant_list.md"
        with open(participants_file, 'w', encoding='utf-8') as f:
            f.write(participants)

        # 保存培训检查清单
        checklist = self.generate_training_checklist()
        checklist_file = training_start_dir / "training_checklist.md"
        with open(checklist_file, 'w', encoding='utf-8') as f:
            f.write(checklist)

        print(f"✅ 培训启动文件已保存到: {training_start_dir}")
        return True

    def run_training_start(self):
        """运行培训启动"""
        print("🚀 开始团队培训启动...")
        print("=" * 60)

        # 检查培训材料
        if not self.check_training_materials():
            print("❌ 培训材料不完整，无法启动培训")
            return False

        # 保存培训启动文件
        self.save_training_files()

        print("\n" + "=" * 60)
        print("🎉 团队培训启动准备完成！")
        print("=" * 60)

        print("\n📋 下一步行动:")
        print("1. 发送培训通知给团队成员")
        print("2. 确认参与者名单")
        print("3. 准备培训环境")
        print("4. 开始第一周培训")
        print("5. 跟踪培训效果")

        print("\n📁 生成的文件:")
        print("- 培训通知: docs/training/startup/training_announcement.md")
        print("- 参与者名单: docs/training/startup/participant_list.md")
        print("- 培训检查清单: docs/training/startup/training_checklist.md")

        return True


def main():
    """主函数"""

    starter = TeamTrainingStarter()
    success = starter.run_training_start()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

# RQA2025 特征层功能增强分析报告（最终部分）

## 2. 功能分析（续）

### 2.5 特征可解释性（续）

#### 2.5.2 特征解释报告

**现状分析**：
缺乏系统化的特征解释报告，难以全面理解特征对模型的影响。

**实现建议**：
实现一个 `FeatureReportGenerator` 类，提供特征解释报告生成功能：

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime
import base64
from io import BytesIO
import jinja2
import markdown

logger = logging.getLogger(__name__)

class FeatureReportGenerator:
    """特征报告生成器"""
    
    def __init__(
        self,
        output_dir: str = './reports',
        feature_explainer: Optional['FeatureExplainer'] = None
    ):
        """
        初始化特征报告生成器
        
        Args:
            output_dir: 输出目录
            feature_explainer: 特征解释器
        """
        self.output_dir = output_dir
        self.feature_explainer = feature_explainer or FeatureExplainer()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载模板
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath='./templates')
        )
    
    def _fig_to_base64(self, fig):
        """
        将matplotlib图形转换为base64编码
        
        Args:
            fig: matplotlib图形
            
        Returns:
            str: base64编码的图形
        """
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str
    
    def generate_feature_importance_section(
        self,
        model,
        X: pd.DataFrame,
        top_n: int = 20
    ) -> Dict:
        """
        生成特征重要性部分
        
        Args:
            model: 模型
            X: 特征数据
            top_n: 显示前N个特征
            
        Returns:
            Dict: 特征重要性部分数据
        """
        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            return {
                'has_importance': False,
                'message': "Model does not have feature_importances_ or coef_ attribute"
            }
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        })
        
        # 排序并获取前N个特征
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(
            x='importance',
            y='feature',
            data=importance_df,
            ax=ax
        )
        ax.set_title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        # 转换图形为base64
        img_str = self._fig_to_base64(fig)
        plt.close(fig)
        
        return {
            'has_importance': True,
            'importance_data': importance_df.to_dict('records'),
            'importance_image': img_str
        }
    
    def generate_feature_correlation_section(
        self,
        X: pd.DataFrame,
        threshold: float = 0.7
    ) -> Dict:
        """
        生成特征相关性部分
        
        Args:
            X: 特征数据
            threshold: 相关性阈值
            
        Returns:
            Dict: 特征相关性部分数据
        """
        # 计算相关性矩阵
        corr_matrix = X.corr()
        
        # 创建热图
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap='coolwarm',
            annot=False,
            fmt='.2f',
            square=True,
            linewidths=.5,
            cbar_kws={'shrink': .5},
            ax=ax
        )
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        
        # 转换图形为base64
        img_str = self._fig_to_base64(fig)
        plt.close(fig)
        
        # 查找高相关性特征对
        high_corr_pairs = []
        
        # 获取上三角矩阵的索引
        upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        
        # 提取上三角矩阵的值
        upper_corr = corr_matrix.where(upper_tri)
        
        # 查找高相关性特征对
        for col in upper_corr.columns:
            for idx, value in upper_corr[col].items():
                if abs(value) > threshold:
                    high_corr_pairs.append({
                        'feature1': idx,
                        'feature2': col,
                        'correlation': value
                    })
        
        # 按相关性绝对值排序
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'correlation_image': img_str,
            'high_correlation_pairs': high_corr_pairs,
            'threshold': threshold
        }
    
    def generate_feature_distribution_section(
        self,
        X: pd.DataFrame,
        max_features: int = 20
    ) -> Dict:
        """
        生成特征分布部分
        
        Args:
            X: 特征数据
            max_features: 最大特征数量
            
        Returns:
            Dict: 特征分布部分数据
        """
        # 选择前N个特征
        if X.shape[1] > max_features:
            # 计算方差
            variances = X.var().sort_values(ascending=False)
            selected_features = variances.index[:max_features]
            X_selected = X[selected_features]
        else:
            X_selected = X
        
        # 创建图形
        n_features = X_selected.shape[1]
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten()
        
        # 绘制分布图
        for i, col in enumerate(X_selected.columns):
            if i < len(axes):
                sns.histplot(X_selected[col], kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel('')
        
        # 隐藏多余的子图
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # 转换图形为base64
        img_str = self._fig_to_base64(fig)
        plt.close(fig)
        
        # 计算统计信息
        stats = X.describe().transpose().reset_index()
        stats = stats.rename(columns={'index': 'feature'})
        
        return {
            'distribution_image': img_str,
            'stats': stats.to_dict('records')
        }
    
    def generate_feature_report(
        self,
        model,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        report_format: str = 'html',
        filename: Optional[str] = None
    ) -> str:
        """
        生成特征报告
        
        Args:
            model: 模型
            X: 特征数据
            y: 目标变量
            report_format: 报告格式，'html'或'markdown'
            filename: 文件名
            
        Returns:
            str: 报告文件路径
        """
        # 生成报告数据
        report_data = {
            'title': 'Feature Analysis Report',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'feature_names': list(X.columns)
            },
            'importance_section': self.generate_feature_importance_section(model, X),
            'correlation_section': self.generate_feature_correlation_section(X),
            'distribution_section': self.generate_feature_distribution_section(X)
        }
        
        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"feature_report_{timestamp}"
        
        # 选择模板
        template_name = 'feature_report_template.html' if report_format == 'html' else 'feature_report_template.md'
        
        try:
            template = self.template_env.get_template(template_name)
            report_content = template.render(**report_data)
        except jinja2.exceptions.TemplateNotFound:
            # 如果模板不存在，则使用内联模板
            if report_format == 'html':
                report_content = self._generate_html_report(report_data)
            else:
                report_content = self._generate_markdown_report(report_data)
        
        # 保存报告
        file_ext = '.html' if report_format == 'html' else '.md'
        report_path = os.path.join(self.output_dir, filename + file_ext)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Feature report saved to {report_path}")
        
        return report_path
    
    def _generate_html_report(self, data: Dict) -> str:
        """
        生成HTML报告
        
        Args:
            data: 报告数据
            
        Returns:
            str: HTML报告内容
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{data['title']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .section {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .image-container {{ margin: 20px 0; }}
                img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <h1>{data['title']}</h1>
            <p>Generated on: {data['timestamp']}</p>
            
            <div class="section">
                <h2>Dataset Information</h2>
                <p>Number of samples: {data['dataset_info']['n_samples']}</p>
                <p>Number of features: {data['dataset_info']['n_features']}</p>
            </div>
        """
        
        # 特征重要性部分
        html += """
            <div class="section">
                <h2>Feature Importance</h2>
        """
        
        if data['importance_section']['has_importance']:
            html += f"""
                <div class="image-container">
                    <img src="data:image/png;base64,{data['importance_section']['importance_image']}" alt="Feature Importance">
                </div>
                <h3>Top Features</h3>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                    </tr>
            """
            
            for item in data['importance_section']['importance_data']:
                html += f"""
                    <tr>
                        <td>{item['feature']}</td>
                        <td>{item['importance']:.4f}</td>
                    </tr>
                """
            
            html += """
                </table>
            """
        else:
            html += f"""
                <p>{data['importance_section']['message']}</p>
            """
        
        html += """
            </div>
        """
        
        # 特征相关性部分
        html += """
            <div class="section">
                <h2>Feature Correlation</h2>
        """
        
        html += f"""
                <div class="image-container">
                    <img src="data:image/png;base64,{data['correlation_section']['correlation_image']}" alt="Feature Correlation">
                </div>
                <h3>High Correlation Pairs (|r| > {data['correlation_section']['threshold']})</h3>
                <table>
                    <tr>
                        <th>Feature 1</th>
                        <th>Feature 2</th>
                        <th>Correlation</th>
                    </tr>
        """
        
        for pair in data['correlation_section']['high_correlation_pairs']:
            html += f"""
                    <tr>
                        <td>{pair['feature1']}</td>
                        <td>{pair['feature2']}</td>
                        <td>{pair['correlation']:.4f}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        """
        
        # 特征分布部分
        html += """
            <div class="section">
                <h2>Feature Distributions</h2>
        """
        
        html += f"""
                <div class="image-container">
                    <img src="data:image/png;base64,{data['distribution_section']['distribution_image']}" alt="Feature
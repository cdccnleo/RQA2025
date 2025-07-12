# RQA2025 模型层功能增强分析报告（续3）

## 2. 功能分析（续）

### 2.2 模型评估（续）

#### 2.2.1 全面评估指标（续）

**实现建议**（续）：

```python
        # 计算MAP
        try:
            ap = average_precision_score(y_true, y_score)
        except Exception as e:
            logger.warning(f"Error calculating MAP: {e}")
            ap = None
        
        # 计算前K个准确率
        top_k_idx = np.argsort(y_score)[::-1][:k]
        precision_at_k = np.mean(y_true[top_k_idx] > 0)
        
        # 计算召回率
        relevant_items = np.where(y_true > 0)[0]
        recall_at_k = len(set(top_k_idx) & set(relevant_items)) / max(1, len(relevant_items))
        
        metrics = {
            'ndcg@k': ndcg,
            'map': ap,
            'precision@k': precision_at_k,
            'recall@k': recall_at_k
        }
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = 'Confusion Matrix',
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'Blues',
        normalize: bool = False,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制混淆矩阵
        
        Args:
            cm: 混淆矩阵
            labels: 标签名称
            title: 图表标题
            figsize: 图形大小
            cmap: 颜色映射
            normalize: 是否归一化
            save_path: 保存路径
            
        Returns:
            plt.Figure: 图形对象
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        
        # 设置标签
        if labels is None:
            labels = [f'Class {i}' for i in range(cm.shape[0])]
        
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels,
            yticklabels=labels,
            title=title,
            ylabel='True label',
            xlabel='Predicted label'
        )
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加文本注释
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black"
                )
        
        fig.tight_layout()
        
        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = 'ROC Curve',
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_prob: 预测概率
            title: 图表标题
            figsize: 图形大小
            save_path: 保存路径
            
        Returns:
            plt.Figure: 图形对象
        """
        from sklearn.metrics import roc_curve, auc
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 处理多分类情况
        if y_prob.shape[1] > 2:
            # 一对多ROC曲线
            y_true_dummies = pd.get_dummies(y_true).values
            
            for i in range(y_prob.shape[1]):
                fpr, tpr, _ = roc_curve(y_true_dummies[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(
                    fpr, tpr,
                    lw=2,
                    label=f'Class {i} (AUC = {roc_auc:.2f})'
                )
        else:
            # 二分类ROC曲线
            prob_col = 1 if y_prob.shape[1] == 2 else 0
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, prob_col])
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr,
                lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})'
            )
        
        # 添加对角线
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        
        ax.set(
            xlim=[0.0, 1.0],
            ylim=[0.0, 1.05],
            xlabel='False Positive Rate',
            ylabel='True Positive Rate',
            title=title
        )
        
        ax.legend(loc="lower right")
        fig.tight_layout()
        
        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = 'Precision-Recall Curve',
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制精确率-召回率曲线
        
        Args:
            y_true: 真实标签
            y_prob: 预测概率
            title: 图表标题
            figsize: 图形大小
            save_path: 保存路径
            
        Returns:
            plt.Figure: 图形对象
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 处理多分类情况
        if y_prob.shape[1] > 2:
            # 一对多PR曲线
            y_true_dummies = pd.get_dummies(y_true).values
            
            for i in range(y_prob.shape[1]):
                precision, recall, _ = precision_recall_curve(y_true_dummies[:, i], y_prob[:, i])
                ap = average_precision_score(y_true_dummies[:, i], y_prob[:, i])
                ax.plot(
                    recall, precision,
                    lw=2,
                    label=f'Class {i} (AP = {ap:.2f})'
                )
        else:
            # 二分类PR曲线
            prob_col = 1 if y_prob.shape[1] == 2 else 0
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, prob_col])
            ap = average_precision_score(y_true, y_prob[:, prob_col])
            ax.plot(
                recall, precision,
                lw=2,
                label=f'PR curve (AP = {ap:.2f})'
            )
        
        ax.set(
            xlim=[0.0, 1.0],
            ylim=[0.0, 1.05],
            xlabel='Recall',
            ylabel='Precision',
            title=title
        )
        
        ax.legend(loc="lower left")
        fig.tight_layout()
        
        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_regression_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = 'Regression Results',
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制回归结果
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
            figsize: 图形大小
            save_path: 保存路径
            
        Returns:
            plt.Figure: 图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 真实值vs预测值散点图
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predictions')
        axes[0, 0].set_title('True vs Predicted Values')
        
        # 残差图
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predictions')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predicted Values')
        
        # 残差分布
        sns.histplot(residuals, kde=True, ax=axes[1, 0])
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        
        # 预测值分布
        sns.histplot(y_pred, kde=True, ax=axes[1, 1], color='green')
        sns.histplot(y_true, kde=True, ax=axes[1, 1], color='blue', alpha=0.5)
        axes[1, 1].set_xlabel('Values')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('True and Predicted Distributions')
        axes[1, 1].legend(['Predicted', 'True'])
        
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_evaluation_report(
        self,
        model_name: str,
        metrics: Dict,
        task_type: str = 'classification',
        plots: Optional[Dict[str, plt.Figure]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        生成评估报告
        
        Args:
            model_name: 模型名称
            metrics: 评估指标
            task_type: 任务类型，'classification'或'regression'或'time_series'或'ranking'
            plots: 图表字典
            save_path: 保存路径
            
        Returns:
            str: 报告内容
        """
        # 创建报告
        report = f"# Model Evaluation Report: {model_name}\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 添加指标
        report += "## Evaluation Metrics\n\n"
        
        if task_type == 'classification':
            report += "### Classification Metrics\n\n"
            report += f"- Accuracy: {metrics.get('accuracy', 'N/A'):.4f}\n"
            report += f"- Precision: {metrics.get('precision', 'N/A'):.4f}\n"
            report += f"- Recall: {metrics.get('recall', 'N/A'):.4f}\n"
            report += f"- F1 Score: {metrics.get('f1', 'N/A'):.4f}\n"
            
            if 'roc_auc' in metrics:
                report += f"- ROC AUC: {metrics['roc_auc']:.4f}\n"
            
            # 添加分类报告
            if 'classification_report' in metrics:
                report += "\n### Detailed Classification Report\n\n"
                report += "```\n"
                cr = metrics['classification_report']
                
                # 创建表头
                report += f"{'':15s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}\n\n"
                
                # 添加每个类别的指标
                for cls, values in cr.items():
                    if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                        report += f"{cls:15s} {values['precision']:10.4f} {values['recall']:10.4f} {values['f1-score']:10.4f} {values['support']:10d}\n"
                
                # 添加平均值
                for avg_type in ['macro avg', 'weighted avg']:
                    if avg_type in cr:
                        report += f"\n{avg_type:15s} {cr[avg_type]['precision']:10.4f} {cr[avg_type]['recall']:10.4f} {cr[avg_type]['f1-score']:10.4f} {cr[avg_type]['support']:10d}\n"
                
                # 添加准确率
                if 'accuracy' in cr:
                    report += f"\n{'accuracy':15s} {''
# 数据层功能增强分析报告（第四部分）

## 功能实现建议（续）

### 3. 监控告警（续）

#### 3.3 数据质量报告（续）

```python
class DataQualityReporter:
    """数据质量报告生成器"""
    
    def __init__(self, report_dir: str = './reports'):
        """
        初始化数据质量报告生成器
        
        Args:
            report_dir: 报告保存目录
        """
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)
    
    def generate_report(
        self,
        quality_data: Dict[str, Any],
        report_format: str = 'json',
        filename: Optional[str] = None
    ) -> str:
        """
        生成数据质量报告
        
        Args:
            quality_data: 数据质量信息
            report_format: 报告格式，支持 'json', 'html', 'markdown'
            filename: 文件名，如果为None则自动生成
            
        Returns:
            str: 报告文件路径
        """
        # 如果没有提供文件名，则自动生成
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data_quality_report_{timestamp}.{report_format}"
        
        filepath = os.path.join(self.report_dir, filename)
        
        # 根据格式生成报告
        if report_format == 'json':
            return self._generate_json_report(quality_data, filepath)
        elif report_format == 'html':
            return self._generate_html_report(quality_data, filepath)
        elif report_format == 'markdown':
            return self._generate_markdown_report(quality_data, filepath)
        else:
            raise ValueError(f"Unsupported report format: {report_format}")
    
    def _generate_json_report(self, quality_data: Dict[str, Any], filepath: str) -> str:
        """
        生成JSON格式报告
        
        Args:
            quality_data: 数据质量信息
            filepath: 文件路径
            
        Returns:
            str: 报告文件路径
        """
        with open(filepath, 'w') as f:
            json.dump(quality_data, f, indent=2)
        return filepath
    
    def _generate_html_report(self, quality_data: Dict[str, Any], filepath: str) -> str:
        """
        生成HTML格式报告
        
        Args:
            quality_data: 数据质量信息
            filepath: 文件路径
            
        Returns:
            str: 报告文件路径
        """
        # 创建HTML报告
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .warning { color: #f39c12; }
                .error { color: #e74c3c; }
                .good { color: #2ecc71; }
            </style>
        </head>
        <body>
            <h1>Data Quality Report</h1>
            <p><strong>Generated:</strong> {timestamp}</p>
            <p><strong>Data Shape:</strong> {rows} rows × {columns} columns</p>
            
            <h2>Missing Values</h2>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Missing Count</th>
                    <th>Missing Ratio</th>
                    <th>Status</th>
                </tr>
        """.format(
            timestamp=quality_data.get('timestamp', datetime.now().isoformat()),
            rows=quality_data.get('data_shape', [0, 0])[0],
            columns=quality_data.get('data_shape', [0, 0])[1]
        )
        
        # 添加缺失值表格
        missing_values = quality_data.get('missing_values', {})
        for column, ratio in missing_values.items():
            status_class = 'good' if ratio < 0.01 else ('warning' if ratio < 0.05 else 'error')
            html += f"""
                <tr>
                    <td>{column}</td>
                    <td>{int(ratio * quality_data.get('data_shape', [0, 0])[0])}</td>
                    <td>{ratio:.2%}</td>
                    <td class="{status_class}">{self._get_status_text(ratio)}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Duplicates</h2>
            <p>Duplicate Rows: {duplicate_count} ({duplicate_ratio:.2%})</p>
            
            <h2>Data Types</h2>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Data Type</th>
                </tr>
        """.format(
            duplicate_count=quality_data.get('duplicates', {}).get('duplicate_count', 0),
            duplicate_ratio=quality_data.get('duplicates', {}).get('duplicate_ratio', 0)
        )
        
        # 添加数据类型表格
        data_types = quality_data.get('data_types', {})
        for column, dtype in data_types.items():
            html += f"""
                <tr>
                    <td>{column}</td>
                    <td>{dtype}</td>
                </tr>
            """
        
        html += """
            </table>
        """
        
        # 添加异常值表格（如果有）
        if 'outliers' in quality_data:
            html += """
            <h2>Outliers</h2>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Outlier Count</th>
                    <th>Outlier Ratio</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std</th>
                </tr>
            """
            
            outliers = quality_data.get('outliers', {})
            for column, info in outliers.items():
                status_class = 'good' if info['outlier_ratio'] < 0.01 else ('warning' if info['outlier_ratio'] < 0.05 else 'error')
                html += f"""
                    <tr>
                        <td>{column}</td>
                        <td>{info['outlier_count']}</td>
                        <td class="{status_class}">{info['outlier_ratio']:.2%}</td>
                        <td>{info['min']:.4g}</td>
                        <td>{info['max']:.4g}</td>
                        <td>{info['mean']:.4g}</td>
                        <td>{info['median']:.4g}</td>
                        <td>{info['std']:.4g}</td>
                    </tr>
                """
            
            html += """
            </table>
            """
        
        # 添加日期范围信息（如果有）
        if 'date_range' in quality_data:
            date_range = quality_data['date_range']
            html += f"""
            <h2>Date Range</h2>
            <p>Start Date: {date_range.get('min_date')}</p>
            <p>End Date: {date_range.get('max_date')}</p>
            <p>Date Range: {date_range.get('date_range_days')} days</p>
            <p>Unique Dates: {date_range.get('date_count')}</p>
            """
        
        # 添加股票代码覆盖率信息（如果有）
        if 'symbol_coverage' in quality_data:
            symbol_coverage = quality_data['symbol_coverage']
            html += f"""
            <h2>Symbol Coverage</h2>
            <p>Symbol Count: {symbol_coverage.get('symbol_count')}</p>
            """
            
            if 'expected_symbol_count' in symbol_coverage:
                coverage_ratio = symbol_coverage.get('coverage_ratio', 0)
                status_class = 'good' if coverage_ratio > 0.95 else ('warning' if coverage_ratio > 0.8 else 'error')
                html += f"""
                <p>Expected Symbol Count: {symbol_coverage.get('expected_symbol_count')}</p>
                <p>Missing Symbol Count: {symbol_coverage.get('missing_symbol_count')}</p>
                <p>Coverage Ratio: <span class="{status_class}">{coverage_ratio:.2%}</span></p>
                """
            
            if 'avg_date_coverage' in symbol_coverage:
                avg_coverage = symbol_coverage.get('avg_date_coverage', 0)
                status_class = 'good' if avg_coverage > 0.95 else ('warning' if avg_coverage > 0.8 else 'error')
                html += f"""
                <p>Average Date Coverage: <span class="{status_class}">{avg_coverage:.2%}</span></p>
                """
        
        html += """
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html)
        
        return filepath
    
    def _generate_markdown_report(self, quality_data: Dict[str, Any], filepath: str) -> str:
        """
        生成Markdown格式报告
        
        Args:
            quality_data: 数据质量信息
            filepath: 文件路径
            
        Returns:
            str: 报告文件路径
        """
        # 创建Markdown报告
        markdown = f"""# Data Quality Report

**Generated:** {quality_data.get('timestamp', datetime.now().isoformat())}

**Data Shape:** {quality_data.get('data_shape', [0, 0])[0]} rows × {quality_data.get('data_shape', [0, 0])[1]} columns

## Missing Values

| Column | Missing Count | Missing Ratio | Status |
|--------|--------------|--------------|--------|
"""
        
        # 添加缺失值表格
        missing_values = quality_data.get('missing_values', {})
        for column, ratio in missing_values.items():
            status = self._get_status_text(ratio)
            markdown += f"| {column} | {int(ratio * quality_data.get('data_shape', [0, 0])[0])} | {ratio:.2%} | {status} |\n"
        
        # 添加重复值信息
        duplicate_count = quality_data.get('duplicates', {}).get('duplicate_count', 0)
        duplicate_ratio = quality_data.get('duplicates', {}).get('duplicate_ratio', 0)
        markdown += f"""
## Duplicates

Duplicate Rows: {duplicate_count} ({duplicate_ratio:.2%})

## Data Types

| Column | Data Type |
|--------|-----------|
"""
        
        # 添加数据类型表格
        data_types = quality_data.get('data_types', {})
        for column, dtype in data_types.items():
            markdown += f"| {column} | {dtype} |\n"
        
        # 添加异常值表格（如果有）
        if 'outliers' in quality_data:
            markdown += """
## Outliers

| Column | Outlier Count | Outlier Ratio | Min | Max | Mean | Median | Std |
|--------|--------------|--------------|-----|-----|------|--------|-----|
"""
            
            outliers = quality_data.get('outliers', {})
            for column, info in outliers.items():
                markdown += f"| {column} | {info['outlier_count']} | {info['outlier_ratio']:.2%} | {info['min']:.4g} | {info['max']:.4g} | {info['mean']:.4g} | {info['median']:.4g} | {info['std']:.4g} |\n"
        
        # 添加日期范围信息（如果有）
        if 'date_range' in quality_data:
            date_range = quality_data['date_range']
            markdown += f"""
## Date Range

- Start Date: {date_range.get('min_date')}
- End Date: {date_range.get('max_date')}
- Date Range: {date_range.get('date_range_days')} days
- Unique Dates: {date_range.get('date_count')}
"""
        
        # 添加股票代码覆盖率信息（如果有）
        if 'symbol_coverage' in quality_data:
            symbol_coverage = quality_data['symbol_coverage']
            markdown += f"""
## Symbol Coverage

- Symbol Count: {symbol_coverage.get('symbol_count')}
"""
            
            if 'expected_symbol_count' in symbol_coverage:
                markdown += f"""
- Expected Symbol Count: {symbol_coverage.get('expected_symbol_count')}
- Missing Symbol Count: {symbol_coverage.get('missing_symbol_count')}
- Coverage Ratio: {symbol_coverage.get('coverage_ratio', 0):.2%}
"""
            
            if 'avg_date_coverage' in symbol_coverage:
                markdown += f"""
- Average Date Coverage: {symbol_coverage.get('avg_date_coverage', 0):.2%}
"""
        
        with open(filepath, 'w') as f:
            f.write(markdown)
        
        return filepath
    
    def _get_status_text(self, ratio: float) -> str:
        """
        根据比例获取状态文本
        
        Args:
            ratio: 比例值
            
        Returns:
            str: 状态文本
        """
        if ratio < 0.01:
            return "Good"
        elif ratio < 0.05:
            return "Warning"
        else:
            return "Critical"
```

在 `DataManager` 中集成数据质量报告功能：

```python
def __init__(self, config: Dict[str, Any]):
    # ... 其他初始化代码 ...
    
    # 初始化数据质量报告生成器
    report_dir = config.get('report_dir', './reports')
    self.quality_reporter = DataQualityReporter(report_dir)

def generate_quality_report(
    self,
    data_model: Optional[DataModel] = None,
    report_format: str = 'html',
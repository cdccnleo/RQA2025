# RQA2025 Web Static Files

## Tailwind CSS 生产环境配置

### ⚠️ 重要提醒

当前页面使用了 Tailwind CSS CDN (`https://cdn.tailwindcss.com`)，这在**开发环境**中是可以的，但在**生产环境**中不推荐使用。

### 🛠️ 生产环境正确配置方法

#### 方法1: 使用 Tailwind CLI (推荐)

1. **安装依赖**
```bash
npm install
# 或
npm install -D tailwindcss
```

2. **构建CSS**
```bash
# 开发模式（监听文件变化）
npm run build-css

# 生产模式（压缩输出）
npm run build-css-prod
```

3. **替换HTML中的CDN引用**
```html
<!-- 移除这一行 -->
<script src="https://cdn.tailwindcss.com"></script>

<!-- 添加本地CSS文件 -->
<link rel="stylesheet" href="./output.css">
```

#### 方法2: 使用 PostCSS

1. **安装 PostCSS 和 Autoprefixer**
```bash
npm install -D postcss autoprefixer
```

2. **创建 PostCSS 配置文件**
```javascript
// postcss.config.js
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  }
}
```

3. **配置构建流程**
```json
// package.json
{
  "scripts": {
    "build": "postcss input.css -o output.css"
  }
}
```

### 📁 文件结构

```
web-static/
├── input.css          # Tailwind 输入文件
├── output.css         # 构建输出文件 (需要添加到 .gitignore)
├── tailwind.config.js # Tailwind 配置文件
├── package.json       # Node.js 依赖配置
├── *.html            # HTML 页面文件
└── README.md         # 本文档
```

### 🚀 部署建议

#### 开发环境
- 可以使用 CDN 快速开发
- 使用 `npm run build-css` 进行实时构建

#### 生产环境
- 必须使用本地构建的 CSS
- 启用压缩和优化
- 考虑使用 CDN 分发静态资源

#### Docker 部署
```dockerfile
# 在 Dockerfile 中构建 CSS
RUN npm install && npm run build-css-prod
```

### 🔧 自定义配置

编辑 `tailwind.config.js` 来自定义：
- 颜色主题
- 字体
- 间距
- 响应式断点
- 动画效果

### 📝 注意事项

1. **文件大小**: 本地构建的 CSS 包含所有使用的类，文件较大但加载更快
2. **缓存策略**: 为 `output.css` 设置适当的缓存头
3. **版本控制**: 将 `output.css` 添加到 `.gitignore`，只提交源文件
4. **性能优化**: 考虑使用 PurgeCSS 移除未使用的 CSS 类

### 🐛 故障排除

**问题**: 样式不生效
**解决**:
1. 检查 `output.css` 是否正确生成
2. 确认 HTML 中的链接路径正确
3. 检查浏览器控制台是否有错误

**问题**: 构建失败
**解决**:
1. 确认 Node.js 和 npm 已安装
2. 检查 `tailwind.config.js` 语法
3. 确认 `input.css` 包含必要的 `@tailwind` 指令

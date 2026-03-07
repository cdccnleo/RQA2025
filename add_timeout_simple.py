# 为strategy-backtest.html添加超时处理机制
# 读取文件
with open(r'c:\PythonProject\RQA2025\web-static\strategy-backtest.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到getApiBaseUrl函数的结束位置
search_str = "        function getApiBaseUrl(endpoint = '') {\n            // 始终使用相对路径，通过nginx代理访问API，确保CORS头正确添加\n            const baseUrl = '/api/v1';\n            return baseUrl + endpoint;\n        }\n\n        let returnsChart"

# 在这个位置之后插入超时处理机制
insert_str = search_str + "\n\n        // API超时处理机制\n        const API_TIMEOUT_MS = 30000;\n        \n        function fetchWithTimeout(url, timeoutMs) {\n            const controller = new AbortController();\n            const id = setTimeout(() => controller.abort(), timeoutMs);\n            return fetch(url, { signal: controller.signal })\n                .then(r => { clearTimeout(id); return r; })\n                .catch(e => ({ ok: false, json: () => Promise.resolve({}), status: 0 }));\n        }\n\n        let returnsChart"

# 替换
new_content = content.replace(search_str, insert_str)

# 写回文件
with open(r'c:\PythonProject\RQA2025\web-static\strategy-backtest.html', 'w', encoding='utf-8') as f:
    f.write(new_content)

print('超时处理机制已添加')
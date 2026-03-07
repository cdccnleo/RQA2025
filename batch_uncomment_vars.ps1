param(
    [string]$Path = "src"
)

Write-Host "开始批量取消未使用变量注释..." -ForegroundColor Green

# 获取所有 Python 文件
$pythonFiles = Get-ChildItem -Path $Path -Filter "*.py" -Recurse
$totalFiles = $pythonFiles.Count
$processedFiles = 0
$totalChanges = 0

Write-Host "找到 $totalFiles 个 Python 文件" -ForegroundColor Yellow

foreach ($file in $pythonFiles) {
    try {
        $content = Get-Content $file.FullName -Raw -Encoding UTF8
        $originalContent = $content

        # 取消注释未使用的变量
        # 处理各种格式的未使用变量注释
        $patterns = @(
            '#\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+?)\s*# 未使用的变量',
            '#\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^#\n]+)\s*# 未使用的变量',
            '#\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(\{.*?\})\s*# 未使用的变量',
            '#\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(\[.*?\])\s*# 未使用的变量',
            '#\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*\(.*?\))\s*# 未使用的变量'
        )

        foreach ($pattern in $patterns) {
            $content = $content -replace $pattern, '$1 = $2'
        }

        if ($content -ne $originalContent) {
            $content | Out-File $file.FullName -Encoding UTF8 -NoNewline
            $changesInFile = ($originalContent | Select-String '# 未使用的变量' -AllMatches).Matches.Count
            $totalChanges += $changesInFile
            $processedFiles++
            Write-Host "处理文件: $($file.FullName) ($changesInFile 处修改)" -ForegroundColor Cyan
        }
    } catch {
        Write-Host "处理文件 $($file.FullName) 时出错: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "`n处理完成:" -ForegroundColor Green
Write-Host "- 处理文件数: $processedFiles" -ForegroundColor Yellow
Write-Host "- 总修改数: $totalChanges" -ForegroundColor Yellow

# 验证修改结果
Write-Host "`n验证修改结果..." -ForegroundColor Green
$remainingComments = Get-ChildItem -Path $Path -Filter "*.py" -Recurse | Where-Object {
    Select-String -Path $_.FullName -Pattern '# 未使用的变量' -Quiet
} | Measure-Object | Select-Object -ExpandProperty Count

if ($remainingComments -eq 0) {
    Write-Host "✓ 所有未使用变量注释已成功取消!" -ForegroundColor Green
} else {
    Write-Host "⚠ 还有 $remainingComments 个文件包含未处理的未使用变量注释" -ForegroundColor Yellow
}

@echo off
echo 开始批量取消未使用变量注释...

REM 使用 PowerShell 来执行 sed 风格的替换
powershell -Command "
$files = Get-ChildItem -Path 'src' -Filter '*.py' -Recurse
$totalFiles = $files.Count
$processedFiles = 0
$totalChanges = 0

Write-Host \"找到 $totalFiles 个 Python 文件\"

foreach ($file in $files) {
    try {
        $content = Get-Content $file.FullName -Raw -Encoding UTF8
        $originalContent = $content

        # 取消注释未使用的变量
        $content = $content -replace '#\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+?)\s*# 未使用的变量', '$1 = $2'

        if ($content -ne $originalContent) {
            $content | Out-File $file.FullName -Encoding UTF8 -NoNewline
            $changesInFile = ($originalContent | Select-String '# 未使用的变量' -AllMatches).Matches.Count
            $totalChanges += $changesInFile
            $processedFiles++
            Write-Host \"处理文件: $($file.FullName) ($changesInFile 处修改)\"
        }
    } catch {
        Write-Host \"处理文件 $($file.FullName) 时出错: $($_.Exception.Message)\"
    }
}

Write-Host \"`n处理完成:\"
Write-Host \"- 处理文件数: $processedFiles\"
Write-Host \"- 总修改数: $totalChanges\"
"

echo 完成！
pause

# Miniconda 修复脚本 - 无备份版本
# 此脚本将直接重新安装 Miniconda（不备份）

Write-Host "=== Miniconda 修复脚本（无备份）===" -ForegroundColor Green
Write-Host "警告: 此操作将直接删除并重新安装 Miniconda，不进行备份!" -ForegroundColor Red

$minicondaPath = "C:\Users\AILeo\miniconda3"

# 1. 直接删除现有Miniconda
Write-Host "`n[1/3] 删除现有Miniconda..." -ForegroundColor Cyan
if (Test-Path $minicondaPath) {
    Write-Host "正在删除: $minicondaPath"
    
    # 尝试使用卸载程序
    $uninstallExe = "$minicondaPath\Uninstall-Miniconda3.exe"
    if (Test-Path $uninstallExe) {
        Write-Host "运行卸载程序..."
        Start-Process -FilePath $uninstallExe -ArgumentList "/S" -Wait
    }
    
    # 强制删除残留
    if (Test-Path $minicondaPath) {
        Write-Host "强制删除目录..."
        Remove-Item -Path $minicondaPath -Recurse -Force -ErrorAction SilentlyContinue
    }
    
    Write-Host "✓ 删除完成"
} else {
    Write-Host "Miniconda目录不存在，跳过删除"
}

# 2. 下载并安装Miniconda
Write-Host "`n[2/3] 下载并安装Miniconda..." -ForegroundColor Cyan
$installerUrl = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
$installerPath = "$env:TEMP\Miniconda3-latest-Windows-x86_64.exe"

Write-Host "下载Miniconda安装程序..."
try {
    # 如果已存在安装程序，先删除
    if (Test-Path $installerPath) {
        Remove-Item -Path $installerPath -Force
    }
    
    Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing
    Write-Host "✓ 下载完成"
    
    Write-Host "安装Miniconda到: $minicondaPath"
    Start-Process -FilePath $installerPath -ArgumentList "/S /D=$minicondaPath" -Wait
    Write-Host "✓ 安装完成"
} catch {
    Write-Host "✗ 下载或安装失败: $_" -ForegroundColor Red
    Write-Host "请手动下载安装: $installerUrl" -ForegroundColor Yellow
    exit 1
}

# 3. 验证安装
Write-Host "`n[3/3] 验证安装..." -ForegroundColor Cyan
$pythonExe = "$minicondaPath\python.exe"
if (Test-Path $pythonExe) {
    try {
        $version = & $pythonExe --version 2>&1
        Write-Host "✓ Python版本: $version" -ForegroundColor Green
        
        if (Test-Path "$minicondaPath\Lib\encodings") {
            Write-Host "✓ 标准库已安装" -ForegroundColor Green
        } else {
            Write-Host "✗ 标准库未安装" -ForegroundColor Red
        }
        
        # 测试conda
        $condaVersion = & "$minicondaPath\Scripts\conda.exe" --version 2>&1
        Write-Host "✓ Conda版本: $condaVersion" -ForegroundColor Green
        
    } catch {
        Write-Host "✗ 验证失败: $_" -ForegroundColor Red
    }
} else {
    Write-Host "✗ Miniconda安装失败，python.exe不存在" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== 修复完成 ===" -ForegroundColor Green
Write-Host "请重新创建项目环境:" -ForegroundColor Cyan
Write-Host "  conda env create -f environment.yml" -ForegroundColor White
Write-Host "  conda activate test" -ForegroundColor White
Write-Host "`n按任意键退出..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

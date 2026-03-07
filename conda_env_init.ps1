# 初始化conda环境变量
$condaPath = 'C:\Users\AILeo\miniconda3\shell\condabin\conda-hook.ps1'
if (Test-Path $condaPath) {
    . $condaPath
}
@echo off
REM 自动更新测试覆盖率面板脚本 (Windows)
REM 用法: auto_update_coverage_dashboard.bat [选项]

setlocal enabledelayedexpansion

REM 配置
set "PROJECT_ROOT=%~dp0.."
set "SCRIPT_DIR=%PROJECT_ROOT%\scripts"
set "LOG_DIR=%PROJECT_ROOT%\logs"
set "REPORTS_DIR=%PROJECT_ROOT%\reports"

REM 默认配置
set UPDATE_INTERVAL=3600
set DB_PATH=data\coverage_monitor.db
set OUTPUT_FILE=reports\coverage_dashboard.html
set LOG_FILE=%LOG_DIR%\coverage_dashboard_auto_update.log
set CONTINUOUS_MODE=false

REM 颜色定义 (Windows CMD)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "RESET=[0m"

REM 日志函数
:log
    echo %DATE% %TIME% - %*
    echo %DATE% %TIME% - %* >> "%LOG_FILE%"
    goto :eof

:error
    echo %RED%%DATE% %TIME% - ERROR: %*%RESET%
    echo %DATE% %TIME% - ERROR: %* >> "%LOG_FILE%"
    goto :eof

:success
    echo %GREEN%%DATE% %TIME% - SUCCESS: %*%RESET%
    echo %DATE% %TIME% - SUCCESS: %* >> "%LOG_FILE%"
    goto :eof

:info
    echo %BLUE%%DATE% %TIME% - INFO: %*%RESET%
    echo %DATE% %TIME% - INFO: %* >> "%LOG_FILE%"
    goto :eof

:warning
    echo %YELLOW%%DATE% %TIME% - WARNING: %*%RESET%
    echo %DATE% %TIME% - WARNING: %* >> "%LOG_FILE%"
    goto :eof

REM 创建必要的目录
:setup_directories
    if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
    if not exist "%REPORTS_DIR%" mkdir "%REPORTS_DIR%"
    for %%I in ("%DB_PATH%") do mkdir "%%~dpI" 2>nul
    goto :eof

REM 更新面板
:update_dashboard
    call :info "开始更新覆盖率面板..."

    if not exist "%DB_PATH%" (
        call :warning "数据库文件不存在: %DB_PATH%"
        call :info "将使用内置数据生成面板..."
    )

    REM 切换到项目根目录并运行脚本
    pushd "%PROJECT_ROOT%"
    python "%SCRIPT_DIR%\generate_coverage_dashboard.py" --db-path "%DB_PATH%" --output "%OUTPUT_FILE%"
    set EXIT_CODE=%errorlevel%
    popd

    if %EXIT_CODE% equ 0 (
        call :success "覆盖率面板更新成功: %OUTPUT_FILE%"
        goto :eof
    ) else (
        call :error "覆盖率面板更新失败"
        goto :eof
    )

REM 显示帮助信息
:show_help
    echo 自动更新测试覆盖率面板脚本 (Windows)
    echo.
    echo 用法:
    echo     %0 [选项]
    echo.
    echo 选项:
    echo     -i, --interval SECONDS    更新间隔(秒)，默认3600(1小时)
    echo     -d, --db-path PATH        数据库路径，默认: data\coverage_monitor.db
    echo     -o, --output PATH         输出文件路径，默认: reports\coverage_dashboard.html
    echo     -l, --log-file PATH       日志文件路径，默认: logs\coverage_dashboard_auto_update.log
    echo     -c, --continuous          连续运行模式
    echo     -s, --single              单次执行模式(默认)
    echo     -h, --help                显示此帮助信息
    echo.
    echo 示例:
    echo     REM 单次更新
    echo     %0 --single
    echo.
    echo     REM 每30分钟自动更新
    echo     %0 --continuous --interval 1800
    echo.
    echo     REM 指定自定义路径
    echo     %0 --db-path C:\path\to\db --output C:\path\to\dashboard.html
    echo.
    echo     REM 在后台运行
    echo     start /B %0 --continuous ^>nul 2^>^&1
    goto :eof

REM 解析命令行参数
:parse_args
    if "%~1"=="" goto :eof

    if "%~1"=="-i" (
        set UPDATE_INTERVAL=%~2
        shift & shift
        goto parse_args
    )
    if "%~1"=="--interval" (
        set UPDATE_INTERVAL=%~2
        shift & shift
        goto parse_args
    )
    if "%~1"=="-d" (
        set DB_PATH=%~2
        shift & shift
        goto parse_args
    )
    if "%~1"=="--db-path" (
        set DB_PATH=%~2
        shift & shift
        goto parse_args
    )
    if "%~1"=="-o" (
        set OUTPUT_FILE=%~2
        shift & shift
        goto parse_args
    )
    if "%~1"=="--output" (
        set OUTPUT_FILE=%~2
        shift & shift
        goto parse_args
    )
    if "%~1"=="-l" (
        set LOG_FILE=%~2
        shift & shift
        goto parse_args
    )
    if "%~1"=="--log-file" (
        set LOG_FILE=%~2
        shift & shift
        goto parse_args
    )
    if "%~1"=="-c" (
        set CONTINUOUS_MODE=true
        shift
        goto parse_args
    )
    if "%~1"=="--continuous" (
        set CONTINUOUS_MODE=true
        shift
        goto parse_args
    )
    if "%~1"=="-s" (
        set CONTINUOUS_MODE=false
        shift
        goto parse_args
    )
    if "%~1"=="--single" (
        set CONTINUOUS_MODE=false
        shift
        goto parse_args
    )
    if "%~1"=="-h" (
        call :show_help
        exit /b 0
    )
    if "%~1"=="--help" (
        call :show_help
        exit /b 0
    )

    call :error "未知选项: %~1"
    call :show_help
    exit /b 1

REM 主函数
:main
    REM 解析参数
    call :parse_args %*

    REM 设置目录
    call :setup_directories

    call :info "=== 覆盖率面板自动更新服务启动 ==="
    call :info "项目根目录: %PROJECT_ROOT%"
    call :info "数据库路径: %DB_PATH%"
    call :info "输出文件: %OUTPUT_FILE%"
    call :info "日志文件: %LOG_FILE%"
    call :info "更新间隔: %UPDATE_INTERVAL%秒"

    if "%CONTINUOUS_MODE%"=="true" (
        call :info "运行模式: 连续更新"
        :continuous_loop
        call :update_dashboard
        if errorlevel 1 (
            call :warning "更新失败，%UPDATE_INTERVAL% 秒后重试..."
        ) else (
            call :info "等待 %UPDATE_INTERVAL% 秒后进行下次更新..."
        )
        timeout /t %UPDATE_INTERVAL% /nobreak >nul
        goto continuous_loop
    ) else (
        call :info "运行模式: 单次执行"
        call :update_dashboard
        if errorlevel 1 (
            exit /b 1
        ) else (
            call :success "单次更新完成"
        )
    )
    goto :eof

REM 执行主函数
call :main %*

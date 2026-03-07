# Phase 1 Infrastructure Deployment Script for RQA2025 Production
# This script orchestrates both server environment preparation and basic services deployment

param(
    [string]$Environment = "production",
    [switch]$SkipEnvironmentPrep,
    [switch]$SkipBasicServices,
    [switch]$SkipHardware,
    [switch]$SkipOS,
    [switch]$SkipNetwork,
    [switch]$SkipSecurity,
    [switch]$SkipDocker,
    [switch]$SkipRedis,
    [switch]$SkipPostgreSQL,
    [switch]$SkipHAProxy,
    [switch]$SkipMonitoring,
    [switch]$ForceRecreate,
    [switch]$GenerateReportOnly,
    [string]$NetworkSubnet = "172.20.0.0/16"
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
$Colors = @{
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "Cyan"
    Header = "Magenta"
}

# Function to write colored output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

# Function to check if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Function to execute server environment preparation
function Invoke-ServerEnvironmentPreparation {
    Write-ColorOutput "Executing Server Environment Preparation..." $Colors.Header
    
    try {
        $scriptPath = Join-Path $PSScriptRoot "prepare_server_environment.ps1"
        if (!(Test-Path $scriptPath)) {
            Write-ColorOutput "Server environment preparation script not found: $scriptPath" $Colors.Error
            return $false
        }
        
        # Build parameters for the script
        $params = @()
        if ($SkipHardware) { $params += "-SkipHardware" }
        if ($SkipOS) { $params += "-SkipOS" }
        if ($SkipNetwork) { $params += "-SkipNetwork" }
        if ($SkipSecurity) { $params += "-SkipSecurity" }
        if ($SkipDocker) { $params += "-SkipDocker" }
        
        $paramString = $params -join " "
        $command = "& '$scriptPath' $paramString"
        
        Write-ColorOutput "Executing: $command" $Colors.Info
        $result = Invoke-Expression $command
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "Server environment preparation completed successfully" $Colors.Success
            return $true
        } else {
            Write-ColorOutput "Server environment preparation failed with exit code: $LASTEXITCODE" $Colors.Error
            return $false
        }
    }
    catch {
        Write-ColorOutput "Server environment preparation execution failed: $($_.Exception.Message)" $Colors.Error
        return $false
    }
}

# Function to execute basic services deployment
function Invoke-BasicServicesDeployment {
    Write-ColorOutput "Executing Basic Services Deployment..." $Colors.Header
    
    try {
        $scriptPath = Join-Path $PSScriptRoot "deploy_basic_services.ps1"
        if (!(Test-Path $scriptPath)) {
            Write-ColorOutput "Basic services deployment script not found: $scriptPath" $Colors.Error
            return $false
        }
        
        # Build parameters for the script
        $params = @()
        $params += "-Environment", $Environment
        if ($SkipRedis) { $params += "-SkipRedis" }
        if ($SkipPostgreSQL) { $params += "-SkipPostgreSQL" }
        if ($SkipHAProxy) { $params += "-SkipHAProxy" }
        if ($SkipMonitoring) { $params += "-SkipMonitoring" }
        if ($ForceRecreate) { $params += "-ForceRecreate" }
        $params += "-NetworkSubnet", $NetworkSubnet
        
        $paramString = $params -join " "
        $command = "& '$scriptPath' $paramString"
        
        Write-ColorOutput "Executing: $command" $Colors.Info
        $result = Invoke-Expression $command
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "Basic services deployment completed successfully" $Colors.Success
            return $true
        } else {
            Write-ColorOutput "Basic services deployment failed with exit code: $LASTEXITCODE" $Colors.Error
            return $false
        }
    }
    catch {
        Write-ColorOutput "Basic services deployment execution failed: $($_.Exception.Message)" $Colors.Error
        return $false
    }
}

# Function to generate comprehensive phase report
function Generate-PhaseReport {
    Write-ColorOutput "Generating Phase 1 Infrastructure Report..." $Colors.Info
    
    $report = @{
        Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Phase = "Phase 1: Infrastructure Preparation"
        Environment = $Environment
        ExecutionSummary = @{
            ServerEnvironmentPrep = if ($SkipEnvironmentPrep) { "Skipped" } else { "Executed" }
            BasicServicesDeployment = if ($SkipBasicServices) { "Skipped" } else { "Executed" }
        }
        Services = @{}
        Checkpoints = @{}
    }
    
    # Check server environment status
    if (!$SkipEnvironmentPrep) {
        try {
            # Check hardware
            if (!$SkipHardware) {
                $cpuCores = (Get-WmiObject -Class Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum
                $memoryGB = [math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
                $diskSpace = [math]::Round((Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace / 1GB, 2)
                
                $report.Checkpoints.Hardware = @{
                    CPU = "$cpuCores cores"
                    Memory = "$memoryGB GB"
                    DiskSpace = "$diskSpace GB free"
                    Status = if ($cpuCores -ge 8 -and $memoryGB -ge 16 -and $diskSpace -ge 100) { "Pass" } else { "Fail" }
                }
            }
            
            # Check OS
            if (!$SkipOS) {
                $os = Get-WmiObject -Class Win32_OperatingSystem
                $report.Checkpoints.OperatingSystem = @{
                    Caption = $os.Caption
                    Version = $os.Version
                    BuildNumber = $os.BuildNumber
                    Status = "Pass"
                }
            }
            
            # Check Docker
            if (!$SkipDocker) {
                try {
                    $dockerInfo = docker info 2>$null
                    $dockerStatus = if ($LASTEXITCODE -eq 0) { "Running" } else { "Not Running" }
                    $report.Checkpoints.Docker = @{
                        Status = $dockerStatus
                        Health = if ($dockerStatus -eq "Running") { "Pass" } else { "Fail" }
                    }
                }
                catch {
                    $report.Checkpoints.Docker = @{
                        Status = "Error"
                        Health = "Fail"
                        Error = $_.Exception.Message
                    }
                }
            }
        }
        catch {
            $report.Checkpoints.Error = "Status check failed: $($_.Exception.Message)"
        }
    }
    
    # Check basic services status
    if (!$SkipBasicServices) {
        try {
            # Check Redis
            if (!$SkipRedis) {
                $redisMasterStatus = docker ps --filter "name=redis-master" --format "{{.Status}}" 2>$null
                $redisSlaveStatus = docker ps --filter "name=redis-slave" --format "{{.Status}}" 2>$null
                
                $report.Services.Redis = @{
                    Master = if ($redisMasterStatus) { "Running" } else { "Not Running" }
                    Slave = if ($redisSlaveStatus) { "Running" } else { "Not Running" }
                    Status = if ($redisMasterStatus -and $redisSlaveStatus) { "Pass" } else { "Fail" }
                }
            }
            
            # Check PostgreSQL
            if (!$SkipPostgreSQL) {
                $postgresMasterStatus = docker ps --filter "name=postgres-master" --format "{{.Status}}" 2>$null
                $postgresSlaveStatus = docker ps --filter "name=postgres-slave" --format "{{.Status}}" 2>$null
                
                $report.Services.PostgreSQL = @{
                    Master = if ($postgresMasterStatus) { "Running" } else { "Not Running" }
                    Slave = if ($postgresSlaveStatus) { "Running" } else { "Not Running" }
                    Status = if ($postgresMasterStatus -and $postgresSlaveStatus) { "Pass" } else { "Fail" }
                }
            }
            
            # Check HAProxy
            if (!$SkipHAProxy) {
                $haproxyStatus = docker ps --filter "name=haproxy" --format "{{.Status}}" 2>$null
                $report.Services.HAProxy = @{
                    Status = if ($haproxyStatus) { "Running" } else { "Not Running" }
                    Health = if ($haproxyStatus) { "Pass" } else { "Fail" }
                }
            }
            
            # Check monitoring services
            if (!$SkipMonitoring) {
                $prometheusStatus = docker ps --filter "name=prometheus" --format "{{.Status}}" 2>$null
                $grafanaStatus = docker ps --filter "name=grafana" --format "{{.Status}}" 2>$null
                
                $report.Services.Monitoring = @{
                    Prometheus = if ($prometheusStatus) { "Running" } else { "Not Running" }
                    Grafana = if ($grafanaStatus) { "Running" } else { "Not Running" }
                    Status = if ($prometheusStatus -and $grafanaStatus) { "Pass" } else { "Fail" }
                }
            }
        }
        catch {
            $report.Services.Error = "Status check failed: $($_.Exception.Message)"
        }
    }
    
    # Calculate overall status
    $allCheckpoints = @()
    $allServices = @()
    
    foreach ($checkpoint in $report.Checkpoints.Values) {
        if ($checkpoint.Status) {
            $allCheckpoints += $checkpoint.Status
        }
    }
    
    foreach ($service in $report.Services.Values) {
        if ($service.Status) {
            $allServices += $service.Status
        }
    }
    
    $overallStatus = "Pass"
    if (($allCheckpoints -contains "Fail") -or ($allServices -contains "Fail")) {
        $overallStatus = "Fail"
    }
    
    $report.OverallStatus = $overallStatus
    
    # Display report
    Write-Host ""
    Write-ColorOutput "===============================================" $Colors.Header
    Write-ColorOutput "Phase 1 Infrastructure Report" $Colors.Header
    Write-ColorOutput "===============================================" $Colors.Header
    Write-ColorOutput "Timestamp: $($report.Timestamp)" $Colors.Info
    Write-ColorOutput "Phase: $($report.Phase)" $Colors.Info
    Write-ColorOutput "Environment: $($report.Environment)" $Colors.Info
    Write-ColorOutput "Overall Status: $overallStatus" -ForegroundColor $(if ($overallStatus -eq "Pass") { $Colors.Success } else { $Colors.Error })
    Write-Host ""
    
    # Display execution summary
    Write-ColorOutput "📋 Execution Summary:" $Colors.Header
    foreach ($item in $report.ExecutionSummary.GetEnumerator()) {
        $color = if ($item.Value -eq "Executed") { $Colors.Success } else { $Colors.Warning }
        Write-ColorOutput "  $($item.Key) - $($item.Value)" -ForegroundColor $color
    }
    Write-Host ""
    
    # Display checkpoints
    if ($report.Checkpoints.Count -gt 0) {
        Write-ColorOutput "📋 Infrastructure Checkpoints:" $Colors.Header
        foreach ($checkpoint in $report.Checkpoints.GetEnumerator()) {
            if ($checkpoint.Key -ne "Error") {
                Write-ColorOutput "  $($checkpoint.Key):" $Colors.Info
                if ($checkpoint.Value -is [hashtable]) {
                    foreach ($item in $checkpoint.Value.GetEnumerator()) {
                        if ($item.Key -eq "Status") {
                            $color = if ($item.Value -eq "Pass") { $Colors.Success } else { $Colors.Error }
                            Write-ColorOutput "    $($item.Key): $($item.Value)" -ForegroundColor $color
                        } else {
                            Write-ColorOutput "    $($item.Key): $($item.Value)" $Colors.Info
                        }
                    }
                }
                Write-Host ""
            }
        }
    }
    
    # Display services
    if ($report.Services.Count -gt 0) {
        Write-ColorOutput "📋 Basic Services:" $Colors.Header
        foreach ($service in $report.Services.GetEnumerator()) {
            if ($service.Key -ne "Error") {
                Write-ColorOutput "  $($service.Key):" $Colors.Info
                if ($service.Value -is [hashtable]) {
                    foreach ($item in $service.Value.GetEnumerator()) {
                        if ($item.Key -eq "Status") {
                            $color = if ($item.Value -eq "Pass") { $Colors.Success } else { $Colors.Error }
                            Write-ColorOutput "    $($item.Key): $($item.Value)" -ForegroundColor $color
                        } else {
                            $color = if ($item.Value -eq "Running") { $Colors.Success } else { $Colors.Warning }
                            Write-ColorOutput "    $($item.Key): $($item.Value)" -ForegroundColor $color
                        }
                    }
                }
                Write-Host ""
            }
        }
    }
    
    # Save report to file
    $reportPath = "deploy\reports\phase1_infrastructure_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
    $reportDir = Split-Path $reportPath -Parent
    if (!(Test-Path $reportDir)) {
        New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
    }
    
    $report | ConvertTo-Json -Depth 10 | Out-File -FilePath $reportPath -Encoding UTF8
    Write-ColorOutput "Phase report saved to: $reportPath" $Colors.Info
    
    return $report
}

# Main execution
function Main {
    Write-ColorOutput "===============================================" $Colors.Header
    Write-ColorOutput "Phase 1: Infrastructure Preparation for RQA2025" $Colors.Header
    Write-ColorOutput "===============================================" $Colors.Header
    Write-Host ""
    
    # Check if running as administrator
    if (!(Test-Administrator)) {
        Write-ColorOutput "This script requires administrator privileges" $Colors.Error
        Write-ColorOutput "Please run PowerShell as Administrator and try again" $Colors.Warning
        return 1
    }
    
    # If only generating report, skip execution
    if ($GenerateReportOnly) {
        Write-ColorOutput "Generate Report Only mode - skipping execution" $Colors.Info
        Generate-PhaseReport
        return 0
    }
    
    # Execute server environment preparation
    if (!$SkipEnvironmentPrep) {
        Write-ColorOutput "Starting Phase 1.1: Server Environment Preparation..." $Colors.Header
        if (!(Invoke-ServerEnvironmentPreparation)) {
            Write-ColorOutput "Server environment preparation failed" $Colors.Error
            return 1
        }
        Write-ColorOutput "Phase 1.1 completed successfully" $Colors.Success
        Write-Host ""
    } else {
        Write-ColorOutput "Skipping server environment preparation" $Colors.Warning
    }
    
    # Execute basic services deployment
    if (!$SkipBasicServices) {
        Write-ColorOutput "Starting Phase 1.2: Basic Services Deployment..." $Colors.Header
        if (!(Invoke-BasicServicesDeployment)) {
            Write-ColorOutput "Basic services deployment failed" $Colors.Error
            return 1
        }
        Write-ColorOutput "Phase 1.2 completed successfully" $Colors.Success
        Write-Host ""
    } else {
        Write-ColorOutput "Skipping basic services deployment" $Colors.Warning
    }
    
    # Generate comprehensive report
    Generate-PhaseReport
    
    Write-Host ""
    Write-ColorOutput "===============================================" $Colors.Header
    Write-ColorOutput "Phase 1 Infrastructure Preparation Complete!" $Colors.Success
    Write-ColorOutput "===============================================" $Colors.Header
    Write-ColorOutput "Next steps:" $Colors.Info
    Write-ColorOutput "1. Review the deployment report" $Colors.Info
    Write-ColorOutput "2. Verify all services are running correctly" $Colors.Info
    Write-ColorOutput "3. Test service connectivity and performance" $Colors.Info
    Write-ColorOutput "4. Proceed to Phase 2: Core Services Deployment" $Colors.Info
    
    return 0
}

# Run main function
try {
    $exitCode = Main
    exit $exitCode
}
catch {
    Write-ColorOutput "Script execution failed: $($_.Exception.Message)" $Colors.Error
    exit 1
}

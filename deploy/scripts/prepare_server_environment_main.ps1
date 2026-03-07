# RQA2025 Server Environment Preparation Main Script v1.0
# Based on Production Deployment Implementation Plan 1.1 Server Environment Preparation
# Usage: .\prepare_server_environment_main.ps1
# Responsible: Operations Team
# Milestone: Infrastructure Ready

param(
    [string]$Environment = "production",
    [switch]$SkipHardwareCheck,
    [switch]$SkipOSCheck,
    [switch]$SkipNetworkCheck,
    [switch]$SkipSecurityCheck,
    [switch]$SkipDockerSetup,
    [switch]$GenerateReportOnly
)

# Global variables
$script:checkpoints = @{}
$script:Environment = $Environment

# Utility functions
function Write-Header {
    param([string]$Message)
    Write-Host "`n" -NoNewline
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Green
}

function Write-Info {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Blue
}

function Write-Warn {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Red
}

# Load environment preparation scripts
function Load-EnvironmentScripts {
    try {
        $scriptPath = Split-Path $MyInvocation.MyCommand.Path
        $serverScript = Join-Path $scriptPath "prepare_server_environment.ps1"
        $dockerScript = Join-Path $scriptPath "prepare_docker_environment.ps1"
        
        if (Test-Path $serverScript) {
            . $serverScript
            Write-Success "Server environment script loaded successfully"
        } else {
            Write-Warn "Server environment script not found: $serverScript"
        }
        
        if (Test-Path $dockerScript) {
            . $dockerScript
            Write-Success "Docker environment script loaded successfully"
        } else {
            Write-Warn "Docker environment script not found: $dockerScript"
        }
        
        return $true
    } catch {
        Write-Error "Failed to load environment scripts: $($_.Exception.Message)"
        return $false
    }
}

# Hardware configuration check
function Invoke-HardwareCheck {
    if ($SkipHardwareCheck) {
        Write-Info "Skipping hardware configuration check"
        $script:checkpoints["Hardware Configuration Complete"] = $true
        return $true
    }
    
    Write-Info "Executing hardware configuration check..."
    
    try {
        # Check CPU cores
        $cpuCores = (Get-WmiObject -Class Win32_Processor).NumberOfCores
        $cpuCheck = $cpuCores -ge 8
        
        # Check memory
        $memoryGB = [math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
        $memoryCheck = $memoryGB -ge 16
        
        # Check disk space
        $diskGB = (Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='C:'").Size / 1GB
        $diskCheck = $diskGB -ge 100
        
        $script:checkpoints["Hardware Configuration Complete"] = ($cpuCheck -and $memoryCheck -and $diskCheck)
        
        if ($script:checkpoints["Hardware Configuration Complete"]) {
            Write-Success "Hardware configuration check passed"
            Write-Info "CPU Cores: $cpuCores, Memory: $memoryGB GB, Disk: $([math]::Round($diskGB, 2)) GB"
        } else {
            Write-Warn "Hardware configuration check failed"
            Write-Info "CPU Cores: $cpuCores (Required: 8+), Memory: $memoryGB GB (Required: 16+), Disk: $([math]::Round($diskGB, 2)) GB (Required: 100+)"
        }
        
        return $script:checkpoints["Hardware Configuration Complete"]
    } catch {
        Write-Error "Hardware configuration check failed: $($_.Exception.Message)"
        $script:checkpoints["Hardware Configuration Complete"] = $false
        return $false
    }
}

# Operating system check
function Invoke-OSCheck {
    if ($SkipOSCheck) {
        Write-Info "Skipping operating system check"
        $script:checkpoints["Operating System Installation Complete"] = $true
        return $true
    }
    
    Write-Info "Executing operating system check..."
    
    try {
        $os = Get-WmiObject -Class Win32_OperatingSystem
        $osName = $os.Caption
        $osVersion = $os.Version
        $osBuild = $os.BuildNumber
        
        # Check if it's Windows 10/11
        $osCheck = $osName -match "Windows 10|Windows 11"
        
        $script:checkpoints["Operating System Installation Complete"] = $osCheck
        
        if ($osCheck) {
            Write-Success "Operating system check passed"
            Write-Info "OS: $osName, Version: $osVersion, Build: $osBuild"
        } else {
            Write-Warn "Operating system check failed"
            Write-Info "Current OS: $osName (Required: Windows 10/11)"
        }
        
        return $osCheck
    } catch {
        Write-Error "Operating system check failed: $($_.Exception.Message)"
        $script:checkpoints["Operating System Installation Complete"] = $false
        return $false
    }
}

# Network configuration check
function Invoke-NetworkCheck {
    if ($SkipNetworkCheck) {
        Write-Info "Skipping network configuration check"
        $script:checkpoints["Network Configuration Complete"] = $true
        return $true
    }
    
    Write-Info "Executing network configuration check..."
    
    try {
        $networkChecks = @()
        
        # Check network adapters
        $adapters = Get-NetAdapter | Where-Object { $_.Status -eq "Up" }
        $networkChecks += ($adapters.Count -gt 0)
        
        # Check IP configuration
        $ipConfig = Get-NetIPAddress | Where-Object { $_.AddressFamily -eq "IPv4" -and $_.IPAddress -notlike "127.*" }
        $networkChecks += ($ipConfig.Count -gt 0)
        
        # Check connectivity
        $loopback = Test-NetConnection -ComputerName "127.0.0.1" -Port 80 -InformationLevel Quiet
        $networkChecks += $loopback
        
        # Check DNS
        $dns = Test-NetConnection -ComputerName "8.8.8.8" -Port 53 -InformationLevel Quiet
        $networkChecks += $dns
        
        # Check external connectivity
        $external = Test-NetConnection -ComputerName "www.baidu.com" -Port 80 -InformationLevel Quiet
        $networkChecks += $external
        
        # Check Windows Firewall
        $firewall = Get-NetFirewallProfile | Where-Object { $_.Enabled -eq $true }
        $networkChecks += ($firewall.Count -gt 0)
        
        # Check required ports
        $ports = @(8000, 5432, 6379, 9090, 3000)
        $portChecks = @()
        foreach ($port in $ports) {
            $portCheck = Test-NetConnection -ComputerName "localhost" -Port $port -InformationLevel Quiet
            $portChecks += $portCheck
        }
        
        $script:checkpoints["Network Configuration Complete"] = ($networkChecks -notcontains $false) -and ($portChecks -notcontains $false)
        
        if ($script:checkpoints["Network Configuration Complete"]) {
            Write-Success "Network configuration check passed"
            Write-Info "Network adapters: $($adapters.Count), IP addresses: $($ipConfig.Count), Required ports available"
        } else {
            Write-Warn "Network configuration check failed"
            Write-Info "Network checks passed: $($networkChecks | Where-Object { $_ -eq $true }).Count/$($networkChecks.Count)"
            Write-Info "Port checks passed: $($portChecks | Where-Object { $_ -eq $true }).Count/$($portChecks.Count)"
        }
        
        return $script:checkpoints["Network Configuration Complete"]
    } catch {
        Write-Error "Network configuration check failed: $($_.Exception.Message)"
        $script:checkpoints["Network Configuration Complete"] = $false
        return $false
    }
}

# Security configuration check
function Invoke-SecurityCheck {
    if ($SkipSecurityCheck) {
        Write-Info "Skipping security configuration check"
        $script:checkpoints["Security Configuration Complete"] = $true
        return $true
    }
    
    Write-Info "Executing security configuration check..."
    
    try {
        $securityChecks = @()
        
        # Check user admin rights
        $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
        $isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
        $securityChecks += $isAdmin
        
        # Check SSH service (if available)
        $sshService = Get-Service -Name "ssh*" -ErrorAction SilentlyContinue
        $securityChecks += ($sshService -ne $null)
        
        # Check Windows Defender
        $defender = Get-MpComputerStatus
        $securityChecks += $defender.AntivirusEnabled
        
        # Check UAC
        $uac = Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System" -Name "EnableLUA" -ErrorAction SilentlyContinue
        $securityChecks += ($uac.EnableLUA -eq 1)
        
        # Check BitLocker
        $bitlocker = Get-BitLockerVolume -ErrorAction SilentlyContinue
        $securityChecks += ($bitlocker -ne $null)
        
        $script:checkpoints["Security Configuration Complete"] = ($securityChecks -notcontains $false)
        
        if ($script:checkpoints["Security Configuration Complete"]) {
            Write-Success "Security configuration check passed"
            Write-Info "Admin rights: $isAdmin, Windows Defender: $($defender.AntivirusEnabled), UAC: $($uac.EnableLUA -eq 1)"
        } else {
            Write-Warn "Security configuration check failed"
            Write-Info "Security checks passed: $($securityChecks | Where-Object { $_ -eq $true }).Count/$($securityChecks.Count)"
        }
        
        return $script:checkpoints["Security Configuration Complete"]
    } catch {
        Write-Error "Security configuration check failed: $($_.Exception.Message)"
        $script:checkpoints["Security Configuration Complete"] = $false
        return $false
    }
}

# Docker environment setup
function Invoke-DockerSetup {
    if ($SkipDockerSetup) {
        Write-Info "Skipping Docker environment setup"
        $script:checkpoints["WSL2 Configuration Complete"] = $true
        $script:checkpoints["Docker Installation Complete"] = $true
        $script:checkpoints["Docker Compose Installation Complete"] = $true
        $script:checkpoints["Docker Service Running Normal"] = $true
        return $true
    }
    
    Write-Info "Executing Docker environment setup..."
    
    try {
        # Check WSL2 support
        $wsl2Result = Test-WSL2Support
        $script:checkpoints["WSL2 Configuration Complete"] = $wsl2Result
        
        # Install Docker Desktop
        $dockerResult = Install-DockerDesktop
        $script:checkpoints["Docker Installation Complete"] = $dockerResult
        
        # Install Docker Compose
        $composeResult = Install-DockerCompose
        $script:checkpoints["Docker Compose Installation Complete"] = $composeResult
        
        # Test Docker environment
        $serviceResult = Test-DockerEnvironment
        $script:checkpoints["Docker Service Running Normal"] = $serviceResult
        
        $overallResult = $wsl2Result -and $dockerResult -and $composeResult -and $serviceResult
        
        if ($overallResult) {
            Write-Success "Docker environment setup completed successfully"
        } else {
            Write-Warn "Docker environment setup completed with issues"
        }
        
        return $overallResult
    } catch {
        Write-Error "Docker environment setup failed: $($_.Exception.Message)"
        $script:checkpoints["WSL2 Configuration Complete"] = $false
        $script:checkpoints["Docker Installation Complete"] = $false
        $script:checkpoints["Docker Compose Installation Complete"] = $false
        $script:checkpoints["Docker Service Running Normal"] = $false
        return $false
    }
}

# Generate comprehensive environment report
function Generate-ComprehensiveReport {
    Write-Header "Environment Preparation Report"
    
    $totalCheckpoints = $script:checkpoints.Count
    $completedCheckpoints = ($script:checkpoints.Values | Where-Object { $_ -eq $true }).Count
    $completionRate = [math]::Round(($completedCheckpoints / $totalCheckpoints) * 100, 2)
    
    Write-Info "Checkpoint completion: $completedCheckpoints/$totalCheckpoints ($completionRate%)"
    Write-Host ""
    
    # Group checkpoints by category
    $categories = @{
        "Basic Environment" = @("Hardware Configuration Complete", "Operating System Installation Complete", "Network Configuration Complete", "Security Configuration Complete")
        "Docker Environment" = @("Docker Installation Complete", "Docker Compose Installation Complete", "WSL2 Configuration Complete", "Docker Service Running Normal")
    }
    
    foreach ($category in $categories.GetEnumerator()) {
        Write-Host "Category: $($category.Key)" -ForegroundColor Yellow
        foreach ($checkpoint in $category.Value) {
            $status = if ($script:checkpoints[$checkpoint]) { "Complete" } else { "Incomplete" }
            $color = if ($script:checkpoints[$checkpoint]) { "Green" } else { "Red" }
            Write-Host "  $checkpoint - $status" -ForegroundColor $color
        }
        Write-Host ""
    }
    
    # Milestone status
    if ($completedCheckpoints -eq $totalCheckpoints) {
        Write-Success "Milestone achieved: Infrastructure Ready"
        Write-Info "1.1 Server Environment Preparation Complete"
        Write-Info "Next step: 1.2 Basic Service Deployment"
        return $true
    } else {
        Write-Warn "Milestone not achieved, need to resolve related issues"
        Write-Info "Incomplete checkpoints:"
        foreach ($checkpoint in $script:checkpoints.GetEnumerator()) {
            if (-not $checkpoint.Value) {
                Write-Host "  - $($checkpoint.Key)" -ForegroundColor Red
            }
        }
        return $false
    }
}

# Save environment report to file
function Save-EnvironmentReport {
    param([string]$ReportPath = "reports\server_environment_report.json")
    
    try {
        $reportDir = Split-Path $ReportPath -Parent
        if (-not (Test-Path $reportDir)) {
            New-Item -ItemType Directory -Path $reportDir -Force
        }
        
        $reportData = @{
            timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            environment = $script:Environment
            checkpoints = $script:checkpoints
            summary = @{
                total = $script:checkpoints.Count
                completed = ($script:checkpoints.Values | Where-Object { $_ -eq $true }).Count
                completion_rate = [math]::Round((($script:checkpoints.Values | Where-Object { $_ -eq $true }).Count / $script:checkpoints.Count) * 100, 2)
            }
            milestone = if (($script:checkpoints.Values | Where-Object { $_ -eq $true }).Count -eq $script:checkpoints.Count) { "Achieved" } else { "Not Achieved" }
            next_step = "1.2 Basic Service Deployment"
        }
        
        $reportData | ConvertTo-Json -Depth 10 | Set-Content $ReportPath
        Write-Success "Environment check report saved: $ReportPath"
        
    } catch {
        Write-Warn "Failed to save environment check report: $($_.Exception.Message)"
    }
}

# Main function
function Main {
    Write-Header "RQA2025 Server Environment Preparation Main Script"
    Write-Info "Based on Production Deployment Implementation Plan 1.1 Server Environment Preparation"
    Write-Info "Responsible: Operations Team | Milestone: Infrastructure Ready"
    Write-Info "Environment: $script:Environment"
    Write-Info "================================================"
    
    $startTime = Get-Date
    
    try {
        # Load scripts
        if (-not (Load-EnvironmentScripts)) {
            Write-Error "Script loading failed, exiting execution"
            return $false
        }
        
        # If only generating report, skip all checks
        if ($GenerateReportOnly) {
            Write-Info "Report generation only mode, skipping environment checks"
            Generate-ComprehensiveReport
            Save-EnvironmentReport
            return $true
        }
        
        # Execute environment checks
        Write-Info "Starting environment checks..."
        
        $hardwareResult = Invoke-HardwareCheck
        $osResult = Invoke-OSCheck
        $networkResult = Invoke-NetworkCheck
        $securityResult = Invoke-SecurityCheck
        $dockerResult = Invoke-DockerSetup
        
        # Generate comprehensive report
        $success = Generate-ComprehensiveReport
        
        # Save report
        Save-EnvironmentReport
        
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        Write-Info "================================================"
        Write-Info "Environment preparation total time: $($duration.TotalSeconds.ToString('F2')) seconds"
        
        if ($success) {
            Write-Success "Milestone achieved: Infrastructure Ready"
            Write-Info "1.1 Server Environment Preparation Complete"
            Write-Info "Next step: 1.2 Basic Service Deployment"
            
            # Show next step suggestions
            Write-Info "Next step suggestions:"
            Write-Info "1. Run basic service deployment script: .\deploy_services.sh"
            Write-Info "2. Verify basic service health status"
            Write-Info "3. Execute infrastructure layer test validation"
            
        } else {
            Write-Warn "Milestone not achieved, need to resolve related issues"
            Write-Info "Problem resolution suggestions:"
            Write-Info "1. Check incomplete checkpoints"
            Write-Info "2. Resolve related issues and re-run script"
            Write-Info "3. Use -SkipXXX parameters to skip completed checks"
        }
        
        return $success
        
    } catch {
        Write-Error "Error occurred during environment preparation: $($_.Exception.Message)"
        Write-Error "Please check error information and re-run script"
        return $false
    }
}

# Show help information
function Show-Help {
    Write-Header "RQA2025 Server Environment Preparation Main Script Help"
    Write-Info "Usage: .\prepare_server_environment_main.ps1 [parameters]"
    Write-Info ""
    Write-Info "Parameter descriptions:"
    Write-Info "  -Environment <environment>     Specify environment (default: production)"
    Write-Info "  -SkipHardwareCheck            Skip hardware configuration check"
    Write-Info "  -SkipOSCheck                  Skip operating system check"
    Write-Info "  -SkipNetworkCheck             Skip network configuration check"
    Write-Info "  -SkipSecurityCheck            Skip security configuration check"
    Write-Info "  -SkipDockerSetup              Skip Docker environment preparation"
    Write-Info "  -GenerateReportOnly           Generate report only, skip checks"
    Write-Info ""
    Write-Info "Examples:"
    Write-Info "  .\prepare_server_environment_main.ps1                    # Complete check"
    Write-Info "  .\prepare_server_environment_main.ps1 -SkipDockerSetup  # Skip Docker"
    Write-Info "  .\prepare_server_environment_main.ps1 -GenerateReportOnly # Generate report only"
}

# Script entry point
if ($MyInvocation.InvocationName -ne '.') {
    # Check help parameters
    if ($args -contains "-h" -or $args -contains "-help" -or $args -contains "--help") {
        Show-Help
        exit 0
    }
    
    # Run main function
    Main
} else {
    # If dot-sourced, only define functions
    Write-Info "Script loaded, can use the following functions:"
    Write-Info "  Load-EnvironmentScripts      - Load environment preparation scripts"
    Write-Info "  Invoke-HardwareCheck         - Execute hardware configuration check"
    Write-Info "  Invoke-OSCheck               - Execute operating system check"
    Write-Info "  Invoke-NetworkCheck          - Execute network configuration check"
    Write-Info "  Invoke-SecurityCheck         - Execute security configuration check"
    Write-Info "  Invoke-DockerSetup           - Execute Docker environment preparation"
    Write-Info "  Generate-ComprehensiveReport - Generate comprehensive environment report"
    Write-Info "  Save-EnvironmentReport       - Save environment check report"
    Write-Info "  Main                         - Execute complete environment preparation"
    Write-Info "  Show-Help                    - Show help information"
}

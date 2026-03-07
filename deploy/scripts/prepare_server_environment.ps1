# RQA2025 Server Environment Preparation Script v1.0
# Based on Production Deployment Implementation Plan 1.1 Server Environment Preparation
# Usage: .\prepare_server_environment.ps1
# Responsible: Operations Team
# Milestone: Infrastructure Ready

# Hardware configuration check
function Test-HardwareConfiguration {
    Write-Info "Checking hardware configuration..."
    
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
        
        $overallCheck = $cpuCheck -and $memoryCheck -and $diskCheck
        
        if ($overallCheck) {
            Write-Success "Hardware configuration check passed"
            Write-Info "CPU Cores: $cpuCores, Memory: $memoryGB GB, Disk: $([math]::Round($diskGB, 2)) GB"
        } else {
            Write-Warn "Hardware configuration check failed"
            Write-Info "CPU Cores: $cpuCores (Required: 8+), Memory: $memoryGB GB (Required: 16+), Disk: $([math]::Round($diskGB, 2)) GB (Required: 100+)"
        }
        
        return $overallCheck
    } catch {
        Write-Error "Hardware configuration check failed: $($_.Exception.Message)"
        return $false
    }
}

# Operating system check
function Test-OperatingSystem {
    Write-Info "Checking operating system..."
    
    try {
        $os = Get-WmiObject -Class Win32_OperatingSystem
        $osName = $os.Caption
        $osVersion = $os.Version
        $osBuild = $os.BuildNumber
        
        # Check if it's Windows 10/11
        $osCheck = $osName -match "Windows 10|Windows 11"
        
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
        return $false
    }
}

# Network configuration check
function Test-NetworkConfiguration {
    Write-Info "Checking network configuration..."
    
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
        
        $overallCheck = ($networkChecks -notcontains $false) -and ($portChecks -notcontains $false)
        
        if ($overallCheck) {
            Write-Success "Network configuration check passed"
            Write-Info "Network adapters: $($adapters.Count), IP addresses: $($ipConfig.Count), Required ports available"
        } else {
            Write-Warn "Network configuration check failed"
            Write-Info "Network checks passed: $($networkChecks | Where-Object { $_ -eq $true }).Count/$($networkChecks.Count)"
            Write-Info "Port checks passed: $($portChecks | Where-Object { $_ -eq $true }).Count/$($portChecks.Count)"
        }
        
        return $overallCheck
    } catch {
        Write-Error "Network configuration check failed: $($_.Exception.Message)"
        return $false
    }
}

# Security configuration check
function Test-SecurityConfiguration {
    Write-Info "Checking security configuration..."
    
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
        
        $overallCheck = ($securityChecks -notcontains $false)
        
        if ($overallCheck) {
            Write-Success "Security configuration check passed"
            Write-Info "Admin rights: $isAdmin, Windows Defender: $($defender.AntivirusEnabled), UAC: $($uac.EnableLUA -eq 1)"
        } else {
            Write-Warn "Security configuration check failed"
            Write-Info "Security checks passed: $($securityChecks | Where-Object { $_ -eq $true }).Count/$($securityChecks.Count)"
        }
        
        return $overallCheck
    } catch {
        Write-Error "Security configuration check failed: $($_.Exception.Message)"
        return $false
    }
}

# Generate environment report
function Generate-EnvironmentReport {
    Write-Header "Server Environment Report"
    
    $checkpoints = @{
        "Hardware Configuration Complete" = Test-HardwareConfiguration
        "Operating System Installation Complete" = Test-OperatingSystem
        "Network Configuration Complete" = Test-NetworkConfiguration
        "Security Configuration Complete" = Test-SecurityConfiguration
    }
    
    $totalCheckpoints = $checkpoints.Count
    $completedCheckpoints = ($checkpoints.Values | Where-Object { $_ -eq $true }).Count
    $completionRate = [math]::Round(($completedCheckpoints / $totalCheckpoints) * 100, 2)
    
    Write-Info "Checkpoint completion: $completedCheckpoints/$totalCheckpoints ($completionRate%)"
    Write-Host ""
    
    foreach ($checkpoint in $checkpoints.GetEnumerator()) {
        $status = if ($checkpoint.Value) { "Complete" } else { "Incomplete" }
        $color = if ($checkpoint.Value) { "Green" } else { "Red" }
        Write-Host "  $($checkpoint.Key) - $status" -ForegroundColor $color
    }
    
    Write-Host ""
    
    if ($completedCheckpoints -eq $totalCheckpoints) {
        Write-Success "All server environment checkpoints completed"
        return $true
    } else {
        Write-Warn "Some server environment checkpoints failed"
        return $false
    }
}

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

# Script entry point
if ($MyInvocation.InvocationName -ne '.') {
    Write-Header "RQA2025 Server Environment Preparation Script"
    Write-Info "Based on Production Deployment Implementation Plan 1.1 Server Environment Preparation"
    Write-Info "Responsible: Operations Team | Milestone: Infrastructure Ready"
    Write-Info "================================================"
    
    Generate-EnvironmentReport
} else {
    Write-Info "Script loaded, can use the following functions:"
    Write-Info "  Test-HardwareConfiguration      - Check hardware configuration"
    Write-Info "  Test-OperatingSystem            - Check operating system"
    Write-Info "  Test-NetworkConfiguration       - Check network configuration"
    Write-Info "  Test-SecurityConfiguration      - Check security configuration"
    Write-Info "  Generate-EnvironmentReport      - Generate comprehensive report"
}

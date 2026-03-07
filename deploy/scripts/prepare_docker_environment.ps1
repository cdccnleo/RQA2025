# Docker Environment Preparation Script for Windows
# This script prepares the Docker environment for RQA2025 production deployment

param(
    [switch]$SkipChecks,
    [switch]$ForceInstall,
    [string]$DockerVersion = "latest"
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

# Function to check Windows version
function Get-WindowsVersion {
    try {
        $os = Get-WmiObject -Class Win32_OperatingSystem
        return @{
            Caption = $os.Caption
            Version = $os.Version
            BuildNumber = $os.BuildNumber
        }
    }
    catch {
        return $null
    }
}

# Function to check if WSL2 is enabled
function Test-WSL2Enabled {
    try {
        $wslStatus = wsl --status 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        return $false
    }
    catch {
        return $false
    }
}

# Function to check if virtualization is enabled
function Test-VirtualizationEnabled {
    try {
        $vmms = Get-WmiObject -Class Msvm_VirtualSystemSettingData -Namespace "root\virtualization\v2" -ErrorAction SilentlyContinue
        return $vmms -ne $null
    }
    catch {
        return $false
    }
}

# Function to check Docker Desktop installation
function Test-DockerDesktopInstalled {
    try {
        $dockerPath = Get-Command docker -ErrorAction SilentlyContinue
        if ($dockerPath) {
            return $true
        }
        
        # Check common installation paths
        $commonPaths = @(
            "${env:ProgramFiles}\Docker\Docker\Docker Desktop.exe",
            "${env:ProgramFiles(x86)}\Docker\Docker\Docker Desktop.exe"
        )
        
        foreach ($path in $commonPaths) {
            if (Test-Path $path) {
                return $true
            }
        }
        
        return $false
    }
    catch {
        return $false
    }
}

# Function to check Docker service status
function Test-DockerServiceRunning {
    try {
        $dockerInfo = docker info 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        return $false
    }
    catch {
        return $false
    }
}

# Function to check Docker Compose
function Test-DockerComposeAvailable {
    try {
        $composeVersion = docker-compose --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        
        # Check Docker Compose V2
        $composeV2 = docker compose version 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        
        return $false
    }
    catch {
        return $false
    }
}

# Function to check available disk space
function Test-DiskSpace {
    param(
        [int]$RequiredGB = 20
    )
    
    try {
        $systemDrive = Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='C:'"
        $freeSpaceGB = [math]::Round($systemDrive.FreeSpace / 1GB, 2)
        $totalSpaceGB = [math]::Round($systemDrive.Size / 1GB, 2)
        
        return @{
            FreeSpaceGB = $freeSpaceGB
            TotalSpaceGB = $totalSpaceGB
            HasEnoughSpace = ($freeSpaceGB -ge $RequiredGB)
        }
    }
    catch {
        return @{
            FreeSpaceGB = 0
            TotalSpaceGB = 0
            HasEnoughSpace = $false
        }
    }
}

# Function to check available memory
function Test-Memory {
    param(
        [int]$RequiredGB = 4
    )
    
    try {
        $memory = Get-WmiObject -Class Win32_ComputerSystem
        $totalMemoryGB = [math]::Round($memory.TotalPhysicalMemory / 1GB, 2)
        
        return @{
            TotalMemoryGB = $totalMemoryGB
            HasEnoughMemory = ($totalMemoryGB -ge $RequiredGB)
        }
    }
    catch {
        return @{
            TotalMemoryGB = 0
            HasEnoughMemory = $false
        }
    }
}

# Function to enable WSL2
function Enable-WSL2 {
    Write-ColorOutput "Enabling WSL2..." $Colors.Info
    
    try {
        # Enable WSL feature
        Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -All -NoRestart
        Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -All -NoRestart
        
        Write-ColorOutput "WSL2 features enabled successfully" $Colors.Success
        Write-ColorOutput "Please restart your computer and run this script again" $Colors.Warning
        return $true
    }
    catch {
        Write-ColorOutput "Failed to enable WSL2: $($_.Exception.Message)" $Colors.Error
        return $false
    }
}

# Function to install Docker Desktop
function Install-DockerDesktop {
    Write-ColorOutput "Installing Docker Desktop..." $Colors.Info
    
    try {
        # Download Docker Desktop installer
        $installerUrl = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
        $installerPath = "$env:TEMP\DockerDesktopInstaller.exe"
        
        Write-ColorOutput "Downloading Docker Desktop installer..." $Colors.Info
        Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath
        
        # Run installer
        Write-ColorOutput "Running Docker Desktop installer..." $Colors.Info
        Start-Process -FilePath $installerPath -ArgumentList "install --quiet" -Wait
        
        # Clean up installer
        Remove-Item $installerPath -Force
        
        Write-ColorOutput "Docker Desktop installed successfully" $Colors.Success
        Write-ColorOutput "Please restart your computer and run this script again" $Colors.Warning
        return $true
    }
    catch {
        Write-ColorOutput "Failed to install Docker Desktop: $($_.Exception.Message)" $Colors.Error
        return $false
    }
}

# Function to start Docker Desktop
function Start-DockerDesktop {
    Write-ColorOutput "Starting Docker Desktop..." $Colors.Info
    
    try {
        Start-Process -FilePath "${env:ProgramFiles}\Docker\Docker\Docker Desktop.exe" -WindowStyle Minimized
        Write-ColorOutput "Docker Desktop started" $Colors.Success
        
        # Wait for Docker to be ready
        Write-ColorOutput "Waiting for Docker to be ready..." $Colors.Info
        $timeout = 120
        $elapsed = 0
        
        while ($elapsed -lt $timeout) {
            if (Test-DockerServiceRunning) {
                Write-ColorOutput "Docker is ready!" $Colors.Success
                return $true
            }
            Start-Sleep -Seconds 5
            $elapsed += 5
            Write-ColorOutput "Still waiting... ($elapsed/$timeout seconds)" $Colors.Info
        }
        
        Write-ColorOutput "Timeout waiting for Docker to be ready" $Colors.Warning
        return $false
    }
    catch {
        Write-ColorOutput "Failed to start Docker Desktop: $($_.Exception.Message)" $Colors.Error
        return $false
    }
}

# Function to test Docker functionality
function Test-DockerFunctionality {
    Write-ColorOutput "Testing Docker functionality..." $Colors.Info
    
    try {
        # Test basic Docker commands
        Write-ColorOutput "Testing docker version..." $Colors.Info
        $dockerVersion = docker version 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "Docker version command failed" $Colors.Error
            return $false
        }
        
        Write-ColorOutput "Testing docker info..." $Colors.Info
        $dockerInfo = docker info 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "Docker info command failed" $Colors.Error
            return $false
        }
        
        Write-ColorOutput "Testing docker run hello-world..." $Colors.Info
        $helloWorld = docker run --rm hello-world 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "Docker hello-world test failed" $Colors.Error
            return $false
        }
        
        Write-ColorOutput "All Docker functionality tests passed!" $Colors.Success
        return $true
    }
    catch {
        Write-ColorOutput "Docker functionality test failed: $($_.Exception.Message)" $Colors.Error
        return $false
    }
}

# Function to configure Docker settings
function Configure-DockerSettings {
    Write-ColorOutput "Configuring Docker settings..." $Colors.Info
    
    try {
        # Create Docker configuration directory
        $dockerConfigDir = "$env:USERPROFILE\.docker"
        if (!(Test-Path $dockerConfigDir)) {
            New-Item -ItemType Directory -Path $dockerConfigDir -Force | Out-Null
        }
        
        # Create daemon.json with recommended settings
        $daemonConfig = @{
            "log-driver" = "json-file"
            "log-opts" = @{
                "max-size" = "10m"
                "max-file" = "3"
            }
            "storage-driver" = "overlay2"
            "experimental" = $false
        }
        
        $daemonConfigPath = "$dockerConfigDir\daemon.json"
        $daemonConfig | ConvertTo-Json -Depth 10 | Set-Content -Path $daemonConfigPath
        
        Write-ColorOutput "Docker daemon configuration created" $Colors.Success
        return $true
    }
    catch {
        Write-ColorOutput "Failed to configure Docker settings: $($_.Exception.Message)" $Colors.Error
        return $false
    }
}

# Main execution
function Main {
    Write-ColorOutput "===============================================" $Colors.Header
    Write-ColorOutput "Docker Environment Preparation for RQA2025" $Colors.Header
    Write-ColorOutput "===============================================" $Colors.Header
    Write-Host ""
    
    # Check if running as administrator
    if (!(Test-Administrator)) {
        Write-ColorOutput "This script requires administrator privileges" $Colors.Error
        Write-ColorOutput "Please run PowerShell as Administrator and try again" $Colors.Warning
        return 1
    }
    
    # Check Windows version
    Write-ColorOutput "Checking Windows version..." $Colors.Info
    $windowsVersion = Get-WindowsVersion
    if ($windowsVersion) {
        Write-ColorOutput "Windows: $($windowsVersion.Caption) (Build $($windowsVersion.BuildNumber))" $Colors.Success
    } else {
        Write-ColorOutput "Failed to get Windows version" $Colors.Warning
    }
    
    # Check system requirements
    Write-ColorOutput "Checking system requirements..." $Colors.Info
    
    $diskSpace = Test-DiskSpace -RequiredGB 20
    Write-ColorOutput "Disk Space: $($diskSpace.FreeSpaceGB)GB free of $($diskSpace.TotalSpaceGB)GB total" -ForegroundColor $(if ($diskSpace.HasEnoughSpace) { $Colors.Success } else { $Colors.Error })
    
    $memory = Test-Memory -RequiredGB 4
    Write-ColorOutput "Memory: $($memory.TotalMemoryGB)GB total" -ForegroundColor $(if ($memory.HasEnoughMemory) { $Colors.Success } else { $Colors.Error })
    
    if (!$diskSpace.HasEnoughSpace -or !$memory.HasEnoughMemory) {
        Write-ColorOutput "System requirements not met" $Colors.Error
        return 1
    }
    
    # Check virtualization
    Write-ColorOutput "Checking virtualization support..." $Colors.Info
    if (Test-VirtualizationEnabled) {
        Write-ColorOutput "Virtualization is enabled" $Colors.Success
    } else {
        Write-ColorOutput "Virtualization is not enabled" $Colors.Error
        Write-ColorOutput "Please enable virtualization in BIOS/UEFI settings" $Colors.Warning
        return 1
    }
    
    # Check WSL2
    Write-ColorOutput "Checking WSL2..." $Colors.Info
    if (Test-WSL2Enabled) {
        Write-ColorOutput "WSL2 is enabled" $Colors.Success
    } else {
        Write-ColorOutput "WSL2 is not enabled" $Colors.Warning
        if ($ForceInstall) {
            if (!(Enable-WSL2)) {
                return 1
            }
        } else {
            Write-ColorOutput "Use -ForceInstall to enable WSL2 automatically" $Colors.Info
            return 1
        }
    }
    
    # Check Docker Desktop installation
    Write-ColorOutput "Checking Docker Desktop installation..." $Colors.Info
    if (Test-DockerDesktopInstalled) {
        Write-ColorOutput "Docker Desktop is installed" $Colors.Success
    } else {
        Write-ColorOutput "Docker Desktop is not installed" $Colors.Warning
        if ($ForceInstall) {
            if (!(Install-DockerDesktop)) {
                return 1
            }
        } else {
            Write-ColorOutput "Use -ForceInstall to install Docker Desktop automatically" $Colors.Info
            return 1
        }
    }
    
    # Check Docker service
    Write-ColorOutput "Checking Docker service..." $Colors.Info
    if (Test-DockerServiceRunning) {
        Write-ColorOutput "Docker service is running" $Colors.Success
    } else {
        Write-ColorOutput "Docker service is not running" $Colors.Warning
        if (!(Start-DockerDesktop)) {
            return 1
        }
    }
    
    # Check Docker Compose
    Write-ColorOutput "Checking Docker Compose..." $Colors.Info
    if (Test-DockerComposeAvailable) {
        Write-ColorOutput "Docker Compose is available" $Colors.Success
    } else {
        Write-ColorOutput "Docker Compose is not available" $Colors.Warning
        Write-ColorOutput "Docker Compose should be included with Docker Desktop" $Colors.Info
    }
    
    # Test Docker functionality
    if (!(Test-DockerFunctionality)) {
        return 1
    }
    
    # Configure Docker settings
    if (!(Configure-DockerSettings)) {
        return 1
    }
    
    Write-Host ""
    Write-ColorOutput "===============================================" $Colors.Header
    Write-ColorOutput "Docker Environment Preparation Complete!" $Colors.Success
    Write-ColorOutput "===============================================" $Colors.Header
    Write-ColorOutput "Next steps:" $Colors.Info
    Write-ColorOutput "1. Verify Docker Desktop is running" $Colors.Info
    Write-ColorOutput "2. Test with: docker run hello-world" $Colors.Info
    Write-ColorOutput "3. Proceed with application deployment" $Colors.Info
    
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

# RQA2025 Production Deployment Script
# PowerShell Version for Windows Environment
# Architecture: 1 Master + 1 Slave (Initial Deployment)
# Usage: .\deploy_production.ps1

param(
    [string]$Environment = "production",
    [string]$ConfigPath = "config\production.yaml"
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Color output functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check dependencies
function Test-Dependencies {
    Write-Info "Checking deployment dependencies..."
    
    # Check Docker
    try {
        $dockerVersion = docker --version
        Write-Info "Docker: $dockerVersion"
    }
    catch {
        Write-Error "Docker is not installed or not in PATH"
        exit 1
    }
    
    # Check Docker Compose
    try {
        $composeVersion = docker-compose --version
        Write-Info "Docker Compose: $composeVersion"
    }
    catch {
        Write-Error "Docker Compose is not installed or not in PATH"
        exit 1
    }
    
    # Check Python
    try {
        $pythonVersion = python --version
        Write-Info "Python: $pythonVersion"
    }
    catch {
        Write-Error "Python is not installed or not in PATH"
        exit 1
    }
    
    Write-Info "All dependencies are available"
}

# Check system requirements
function Test-SystemRequirements {
    Write-Info "Checking system requirements..."
    
    # Check D drive availability
    if (!(Test-Path "D:\")) {
        Write-Error "D drive is not available"
        exit 1
    }
    
    # Check available memory
    $memory = Get-WmiObject -Class Win32_ComputerSystem | Select-Object -ExpandProperty TotalPhysicalMemory
    $memoryGB = [math]::Round($memory / 1GB, 2)
    Write-Info "Total Memory: $memoryGB GB"
    
    if ($memoryGB -lt 8) {
        Write-Warn "Memory is less than 8GB recommended"
    }
    
    # Check available disk space on D drive
    $disk = Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='D:'" | Select-Object -ExpandProperty FreeSpace
    $diskGB = [math]::Round($disk / 1GB, 2)
    Write-Info "Available Disk Space on D: $diskGB GB"
    
    if ($diskGB -lt 50) {
        Write-Warn "Disk space on D: is less than 50GB recommended"
    }
    
    # Check CPU cores
    $cpuCores = (Get-WmiObject -Class Win32_Processor).NumberOfCores
    Write-Info "CPU Cores: $cpuCores"
    
    if ($cpuCores -lt 4) {
        Write-Warn "CPU cores are less than 4 recommended"
    }
}

# Create directories on D drive
function New-Directories {
    Write-Info "Creating necessary directories on D drive..."
    
    $directories = @(
        "D:\rqa2025\config",
        "D:\rqa2025\logs", 
        "D:\rqa2025\data",
        "D:\rqa2025\models",
        "D:\rqa2025\cache",
        "D:\rqa2025\logs\app",
        "D:\rqa2025\config\app",
        "D:\rqa2025\monitoring\prometheus",
        "D:\rqa2025\monitoring\grafana",
        "D:\rqa2025\monitoring\alertmanager",
        "D:\rqa2025\logging\elasticsearch",
        "D:\rqa2025\logging\logstash",
        "D:\rqa2025\logging\kibana",
        "D:\rqa2025\backup\database",
        "D:\rqa2025\backup\logs",
        "D:\rqa2025\backup\models",
        "D:\rqa2025\data\postgres-master",
        "D:\rqa2025\data\postgres-slave",
        "D:\rqa2025\cache\redis-master",
        "D:\rqa2025\cache\redis-slave"
    )
    
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Info "Created directory: $dir"
        }
    }
}

# Build Docker images
function Build-DockerImages {
    Write-Info "Building Docker images..."
    
    # Build API image
    Write-Info "Building RQA2025 API image..."
    docker build -f ..\Dockerfile -t rqa2025/api:latest ..\..
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build API image"
        exit 1
    }
    
    # Build inference image
    Write-Info "Building inference engine image..."
    docker build -f ..\Dockerfile.inference -t rqa2025/inference:latest ..\..
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build inference image"
        exit 1
    }
    
    Write-Info "Docker images built successfully"
}

# Deploy services
function Deploy-Services {
    Write-Info "Deploying services (1 Master + 1 Slave architecture)..."
    
    # Start PostgreSQL Master
    Write-Info "Starting PostgreSQL Master..."
    docker-compose -f ..\docker-compose.yml up -d postgres-master
    Start-Sleep -Seconds 30
    
    # Start PostgreSQL Slave
    Write-Info "Starting PostgreSQL Slave..."
    docker-compose -f ..\docker-compose.yml up -d postgres-slave
    Start-Sleep -Seconds 30
    
    # Start Redis Master
    Write-Info "Starting Redis Master..."
    docker-compose -f ..\docker-compose.yml up -d redis-master
    Start-Sleep -Seconds 30
    
    # Start Redis Slave
    Write-Info "Starting Redis Slave..."
    docker-compose -f ..\docker-compose.yml up -d redis-slave
    Start-Sleep -Seconds 30
    
    # Start monitoring services
    Write-Info "Starting monitoring services..."
    docker-compose -f ..\docker-compose.yml up -d prometheus grafana alertmanager
    Start-Sleep -Seconds 30
    
    # Start logging services
    Write-Info "Starting logging services..."
    docker-compose -f ..\docker-compose.yml up -d elasticsearch logstash kibana
    Start-Sleep -Seconds 60
    
    # Start application services - Master
    Write-Info "Starting application services (Master)..."
    docker-compose -f ..\docker-compose.yml up -d rqa2025-api-master inference-engine-master
    Start-Sleep -Seconds 30
    
    # Start application services - Slave
    Write-Info "Starting application services (Slave)..."
    docker-compose -f ..\docker-compose.yml up -d rqa2025-api-slave inference-engine-slave
    Start-Sleep -Seconds 30
    
    # Start load balancer
    Write-Info "Starting load balancer..."
    docker-compose -f ..\docker-compose.yml up -d nginx
    Start-Sleep -Seconds 30
    
    Write-Info "All services deployed successfully (1 Master + 1 Slave)"
}

# Health check
function Test-HealthCheck {
    Write-Info "Performing health checks..."
    
    $maxRetries = 10
    $retryCount = 0
    
    while ($retryCount -lt $maxRetries) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost/health" -TimeoutSec 10
            if ($response.StatusCode -eq 200) {
                Write-Info "Health check passed"
                return $true
            }
        }
        catch {
            Write-Warn "Health check failed, retrying... ($($retryCount + 1)/$maxRetries)"
        }
        
        $retryCount++
        Start-Sleep -Seconds 10
    }
    
    Write-Error "Health check failed after $maxRetries attempts"
    return $false
}

# Verify functionality
function Test-Functionality {
    Write-Info "Verifying functionality..."
    
    # Test API endpoints
    $endpoints = @(
        "http://localhost/health",
        "http://localhost/api/v1/status",
        "http://localhost/metrics"
    )
    
    foreach ($endpoint in $endpoints) {
        try {
            $response = Invoke-WebRequest -Uri $endpoint -TimeoutSec 10
            Write-Info "API endpoint $endpoint is working"
        }
        catch {
            Write-Warn "API endpoint $endpoint is not responding"
        }
    }
    
    # Test monitoring
    try {
        $prometheusResponse = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/targets" -TimeoutSec 10
        Write-Info "Prometheus is working"
    }
    catch {
        Write-Warn "Prometheus is not responding"
    }
    
    try {
        $grafanaResponse = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -TimeoutSec 10
        Write-Info "Grafana is working"
    }
    catch {
        Write-Warn "Grafana is not responding"
    }
}

# Monitor deployment
function Start-Monitoring {
    Write-Info "Starting deployment monitoring..."
    
    # Check service status
    $services = docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    Write-Info "Service Status:"
    Write-Host $services
    
    # Check resource usage
    $stats = docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
    Write-Info "Resource Usage:"
    Write-Host $stats
}

# Generate deployment report
function New-DeploymentReport {
    Write-Info "Generating deployment report..."
    
    $reportPath = "D:\rqa2025\deployment_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
    
    $report = @"
RQA2025 Production Deployment Report
====================================
Deployment Time: $(Get-Date)
Environment: $Environment
Deployment Path: D:\rqa2025
Architecture: 1 Master + 1 Slave

Deployment Status: SUCCESS

Services Deployed:
- RQA2025 API Service Master (1 replica)
- RQA2025 API Service Slave (1 replica)
- Inference Engine Master (1 replica)
- Inference Engine Slave (1 replica)
- PostgreSQL Database Master
- PostgreSQL Database Slave
- Redis Master
- Redis Slave
- Nginx Load Balancer
- Prometheus Monitoring
- Grafana Dashboard
- AlertManager
- Elasticsearch
- Logstash
- Kibana

Access URLs:
- API Gateway: http://localhost
- API Master: http://localhost:8000
- API Slave: http://localhost:8002
- Inference Master: http://localhost:8001
- Inference Slave: http://localhost:8003
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Kibana: http://localhost:5601

Configuration Files:
- Application Config: D:\rqa2025\config
- Logs: D:\rqa2025\logs
- Data: D:\rqa2025\data
- Models: D:\rqa2025\models
- Monitoring: D:\rqa2025\monitoring
- Logging: D:\rqa2025\logging
- Backup: D:\rqa2025\backup

Database Configuration:
- Master: localhost:5432
- Slave: localhost:5433

Cache Configuration:
- Master: localhost:6379
- Slave: localhost:6380

Next Steps:
1. Configure SSL certificates
2. Set up backup schedules
3. Configure alert notifications
4. Monitor system performance
5. Test all business functions
6. Configure master-slave replication

"@
    
    $report | Out-File -FilePath $reportPath -Encoding UTF8
    Write-Info "Deployment report saved to: $reportPath"
}

# Main deployment function
function Start-Deployment {
    Write-Info "Starting RQA2025 production deployment on D drive..."
    Write-Info "Environment: $Environment"
    Write-Info "Config Path: $ConfigPath"
    Write-Info "Deployment Path: D:\rqa2025"
    Write-Info "Architecture: 1 Master + 1 Slave"
    
    # Phase 1: Infrastructure preparation
    Write-Info "Phase 1: Infrastructure preparation"
    Test-Dependencies
    Test-SystemRequirements
    New-Directories
    
    # Phase 2: Build and deploy
    Write-Info "Phase 2: Building and deploying services"
    Build-DockerImages
    Deploy-Services
    
    # Phase 3: Verification
    Write-Info "Phase 3: Verifying deployment"
    if (Test-HealthCheck) {
        Test-Functionality
        Start-Monitoring
        New-DeploymentReport
        
        Write-Info "Deployment completed successfully!"
        Write-Info "Services are available at:"
        Write-Info "  - API Gateway: http://localhost"
        Write-Info "  - API Master: http://localhost:8000"
        Write-Info "  - API Slave: http://localhost:8002"
        Write-Info "  - Inference Master: http://localhost:8001"
        Write-Info "  - Inference Slave: http://localhost:8003"
        Write-Info "  - Prometheus: http://localhost:9090"
        Write-Info "  - Grafana: http://localhost:3000"
        Write-Info "  - Kibana: http://localhost:5601"
        Write-Info ""
        Write-Info "All data is stored on D:\rqa2025"
        Write-Info "Configuration files: D:\rqa2025\config"
        Write-Info "Log files: D:\rqa2025\logs"
        Write-Info "Backup files: D:\rqa2025\backup"
        Write-Info ""
        Write-Info "Architecture: 1 Master + 1 Slave"
        Write-Info "Database Master: localhost:5432"
        Write-Info "Database Slave: localhost:5433"
        Write-Info "Redis Master: localhost:6379"
        Write-Info "Redis Slave: localhost:6380"
    }
    else {
        Write-Error "Deployment failed - health check did not pass"
        exit 1
    }
}

# Execute deployment
try {
    Start-Deployment
}
catch {
    Write-Error "Deployment failed: $($_.Exception.Message)"
    exit 1
} 
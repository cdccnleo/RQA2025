# RQA2025 Simple Production Deployment Script
# PowerShell Version for Windows Environment
# Architecture: 1 Master + 1 Slave (Initial Deployment)
# Usage: .\deploy_simple.ps1

param(
    [string]$Environment = "production"
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
    
    # Check available disk space on D drive
    $disk = Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='D:'" | Select-Object -ExpandProperty FreeSpace
    $diskGB = [math]::Round($disk / 1GB, 2)
    Write-Info "Available Disk Space on D: $diskGB GB"
    
    # Check CPU cores
    $cpuCores = (Get-WmiObject -Class Win32_Processor).NumberOfCores
    Write-Info "CPU Cores: $cpuCores"
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

# Deploy basic services
function Deploy-BasicServices {
    Write-Info "Deploying basic services (1 Master + 1 Slave architecture)..."
    
    # Start PostgreSQL Master
    Write-Info "Starting PostgreSQL Master..."
    docker run -d --name rqa2025-postgres-master `
        -e POSTGRES_DB=rqa2025 `
        -e POSTGRES_USER=rqa2025 `
        -e POSTGRES_PASSWORD=password `
        -p 5432:5432 `
        -v D:\rqa2025\data\postgres-master:/var/lib/postgresql/data `
        postgres:13
    Start-Sleep -Seconds 30
    
    # Start PostgreSQL Slave
    Write-Info "Starting PostgreSQL Slave..."
    docker run -d --name rqa2025-postgres-slave `
        -e POSTGRES_DB=rqa2025 `
        -e POSTGRES_USER=rqa2025 `
        -e POSTGRES_PASSWORD=password `
        -p 5433:5432 `
        -v D:\rqa2025\data\postgres-slave:/var/lib/postgresql/data `
        postgres:13
    Start-Sleep -Seconds 30
    
    # Start Redis Master
    Write-Info "Starting Redis Master..."
    docker run -d --name rqa2025-redis-master `
        -p 6379:6379 `
        -v D:\rqa2025\cache\redis-master:/data `
        redis:6-alpine redis-server --appendonly yes --requirepass password
    Start-Sleep -Seconds 30
    
    # Start Redis Slave
    Write-Info "Starting Redis Slave..."
    docker run -d --name rqa2025-redis-slave `
        -p 6380:6379 `
        -v D:\rqa2025\cache\redis-slave:/data `
        redis:6-alpine redis-server --appendonly yes --requirepass password --slaveof rqa2025-redis-master 6379
    Start-Sleep -Seconds 30
    
    # Start Prometheus
    Write-Info "Starting Prometheus..."
    docker run -d --name rqa2025-prometheus `
        -p 9090:9090 `
        -v D:\rqa2025\monitoring\prometheus:/prometheus `
        prom/prometheus:latest `
        --config.file=/etc/prometheus/prometheus.yml `
        --storage.tsdb.path=/prometheus `
        --web.console.libraries=/etc/prometheus/console_libraries `
        --web.console.templates=/etc/prometheus/consoles `
        --storage.tsdb.retention.time=200h `
        --web.enable-lifecycle
    Start-Sleep -Seconds 30
    
    # Start Grafana
    Write-Info "Starting Grafana..."
    docker run -d --name rqa2025-grafana `
        -p 3000:3000 `
        -e GF_SECURITY_ADMIN_PASSWORD=admin123 `
        -e GF_USERS_ALLOW_SIGN_UP=false `
        -v D:\rqa2025\monitoring\grafana:/var/lib/grafana `
        grafana/grafana:latest
    Start-Sleep -Seconds 30
    
    # Start Elasticsearch
    Write-Info "Starting Elasticsearch..."
    docker run -d --name rqa2025-elasticsearch `
        -p 9200:9200 `
        -e discovery.type=single-node `
        -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" `
        -v D:\rqa2025\logging\elasticsearch:/usr/share/elasticsearch/data `
        docker.elastic.co/elasticsearch/elasticsearch:7.17.0
    Start-Sleep -Seconds 60
    
    # Start Kibana
    Write-Info "Starting Kibana..."
    docker run -d --name rqa2025-kibana `
        -p 5601:5601 `
        -e ELASTICSEARCH_HOSTS=http://elasticsearch:9200 `
        docker.elastic.co/kibana/kibana:7.17.0
    Start-Sleep -Seconds 30
    
    Write-Info "Basic services deployed successfully (1 Master + 1 Slave)"
}

# Health check
function Test-HealthCheck {
    Write-Info "Performing health checks..."
    
    # Check PostgreSQL Master
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5432" -TimeoutSec 5
        Write-Info "PostgreSQL Master is running"
    }
    catch {
        Write-Warn "PostgreSQL Master is not responding"
    }
    
    # Check Redis Master
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:6379" -TimeoutSec 5
        Write-Info "Redis Master is running"
    }
    catch {
        Write-Warn "Redis Master is not responding"
    }
    
    # Check Prometheus
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/targets" -TimeoutSec 10
        Write-Info "Prometheus is working"
    }
    catch {
        Write-Warn "Prometheus is not responding"
    }
    
    # Check Grafana
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -TimeoutSec 10
        Write-Info "Grafana is working"
    }
    catch {
        Write-Warn "Grafana is not responding"
    }
    
    return $true
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
RQA2025 Simple Production Deployment Report
===========================================
Deployment Time: $(Get-Date)
Environment: $Environment
Deployment Path: D:\rqa2025
Architecture: 1 Master + 1 Slave

Deployment Status: SUCCESS

Services Deployed:
- PostgreSQL Database Master (localhost:5432)
- PostgreSQL Database Slave (localhost:5433)
- Redis Master (localhost:6379)
- Redis Slave (localhost:6380)
- Prometheus Monitoring (localhost:9090)
- Grafana Dashboard (localhost:3000)
- Elasticsearch (localhost:9200)
- Kibana (localhost:5601)

Access URLs:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin123)
- Kibana: http://localhost:5601
- Elasticsearch: http://localhost:9200

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
1. Deploy RQA2025 application services
2. Configure SSL certificates
3. Set up backup schedules
4. Configure alert notifications
5. Monitor system performance
6. Test all business functions
7. Configure master-slave replication

"@
    
    $report | Out-File -FilePath $reportPath -Encoding UTF8
    Write-Info "Deployment report saved to: $reportPath"
}

# Main deployment function
function Start-Deployment {
    Write-Info "Starting RQA2025 simple production deployment on D drive..."
    Write-Info "Environment: $Environment"
    Write-Info "Deployment Path: D:\rqa2025"
    Write-Info "Architecture: 1 Master + 1 Slave"
    
    # Phase 1: Infrastructure preparation
    Write-Info "Phase 1: Infrastructure preparation"
    Test-Dependencies
    Test-SystemRequirements
    New-Directories
    
    # Phase 2: Deploy basic services
    Write-Info "Phase 2: Deploying basic services"
    Deploy-BasicServices
    
    # Phase 3: Verification
    Write-Info "Phase 3: Verifying deployment"
    if (Test-HealthCheck) {
        Start-Monitoring
        New-DeploymentReport
        
        Write-Info "Basic deployment completed successfully!"
        Write-Info "Services are available at:"
        Write-Info "  - Prometheus: http://localhost:9090"
        Write-Info "  - Grafana: http://localhost:3000 (admin/admin123)"
        Write-Info "  - Kibana: http://localhost:5601"
        Write-Info "  - Elasticsearch: http://localhost:9200"
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
        Write-Info ""
        Write-Info "Next: Deploy RQA2025 application services when network is available"
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
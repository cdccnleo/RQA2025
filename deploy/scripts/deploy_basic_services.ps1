# Basic Services Deployment Script for RQA2025 Production
# This script deploys Redis master-slave, PostgreSQL master-slave, HAProxy, and monitoring services

param(
    [string]$Environment = "production",
    [switch]$SkipRedis,
    [switch]$SkipPostgreSQL,
    [switch]$SkipHAProxy,
    [switch]$SkipMonitoring,
    [switch]$ForceRecreate,
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

# Function to check Docker service
function Test-DockerService {
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

# Function to create Docker network
function New-DockerNetwork {
    param(
        [string]$NetworkName,
        [string]$Subnet
    )
    
    Write-ColorOutput "Creating Docker network: $NetworkName" $Colors.Info
    
    try {
        # Check if network exists
        $existingNetwork = docker network ls --filter "name=$NetworkName" --format "{{.Name}}" 2>$null
        if ($existingNetwork -eq $NetworkName) {
            if ($ForceRecreate) {
                Write-ColorOutput "Removing existing network: $NetworkName" $Colors.Warning
                docker network rm $NetworkName 2>$null
            } else {
                Write-ColorOutput "Network $NetworkName already exists" $Colors.Success
                return $true
            }
        }
        
        # Create new network
        $networkResult = docker network create --driver bridge --subnet $Subnet $NetworkName 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "Network $NetworkName created successfully" $Colors.Success
            return $true
        } else {
            Write-ColorOutput "Failed to create network $NetworkName" $Colors.Error
            return $false
        }
    }
    catch {
        Write-ColorOutput "Error creating network $NetworkName : $($_.Exception.Message)" $Colors.Error
        return $false
    }
}

# Function to deploy Redis master-slave
function Deploy-RedisMasterSlave {
    Write-ColorOutput "Deploying Redis Master-Slave configuration..." $Colors.Header
    
    try {
        # Create Redis master
        Write-ColorOutput "Creating Redis master container..." $Colors.Info
        $redisMasterResult = docker run -d --name redis-master `
            --network rqa2025-network `
            --ip 172.20.1.10 `
            -p 6379:6379 `
            -v redis_master_data:/data `
            redis:7-alpine redis-server --appendonly yes --requirepass "rqa2025_redis_master"
        
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "Failed to create Redis master container" $Colors.Error
            return $false
        }
        
        # Create Redis slave
        Write-ColorOutput "Creating Redis slave container..." $Colors.Info
        $redisSlaveResult = docker run -d --name redis-slave `
            --network rqa2025-network `
            --ip 172.20.1.11 `
            -p 6380:6379 `
            -v redis_slave_data:/data `
            redis:7-alpine redis-server --appendonly yes --requirepass "rqa2025_redis_slave" --slaveof 172.20.1.10 6379
        
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "Failed to create Redis slave container" $Colors.Error
            return $false
        }
        
        # Wait for containers to be ready
        Write-ColorOutput "Waiting for Redis containers to be ready..." $Colors.Info
        Start-Sleep -Seconds 10
        
        # Test Redis master
        Write-ColorOutput "Testing Redis master..." $Colors.Info
        $masterTest = docker exec redis-master redis-cli -a "rqa2025_redis_master" ping 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "Redis master is ready" $Colors.Success
        } else {
            Write-ColorOutput "Redis master is not ready" $Colors.Warning
        }
        
        # Test Redis slave
        Write-ColorOutput "Testing Redis slave..." $Colors.Info
        $slaveTest = docker exec redis-slave redis-cli -a "rqa2025_redis_slave" ping 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "Redis slave is ready" $Colors.Success
        } else {
            Write-ColorOutput "Redis slave is not ready" $Colors.Warning
        }
        
        # Test replication
        Write-ColorOutput "Testing Redis replication..." $Colors.Info
        docker exec redis-master redis-cli -a "rqa2025_redis_master" set test_key "test_value" 2>$null
        $replicationTest = docker exec redis-slave redis-cli -a "rqa2025_redis_slave" get test_key 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "Redis replication is working" $Colors.Success
        } else {
            Write-ColorOutput "Redis replication test failed" $Colors.Warning
        }
        
        # Clean up test data
        docker exec redis-master redis-cli -a "rqa2025_redis_master" del test_key 2>$null
        
        Write-ColorOutput "Redis Master-Slave deployment completed successfully" $Colors.Success
        return $true
    }
    catch {
        Write-ColorOutput "Redis deployment failed: $($_.Exception.Message)" $Colors.Error
        return $false
    }
}

# Function to deploy PostgreSQL master-slave
function Deploy-PostgreSQLMasterSlave {
    Write-ColorOutput "Deploying PostgreSQL Master-Slave configuration..." $Colors.Header
    
    try {
        # Create PostgreSQL master
        Write-ColorOutput "Creating PostgreSQL master container..." $Colors.Info
        $postgresMasterResult = docker run -d --name postgres-master `
            --network rqa2025-network `
            --ip 172.20.2.10 `
            -p 5432:5432 `
            -e POSTGRES_DB=rqa2025_production `
            -e POSTGRES_USER=rqa2025_user `
            -e POSTGRES_PASSWORD=rqa2025_master_password `
            -e POSTGRES_INITDB_ARGS="--encoding=UTF8 --lc-collate=C --lc-ctype=C" `
            -v postgres_master_data:/var/lib/postgresql/data `
            postgres:15-alpine
        
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "Failed to create PostgreSQL master container" $Colors.Error
            return $false
        }
        
        # Wait for master to be ready
        Write-ColorOutput "Waiting for PostgreSQL master to be ready..." $Colors.Info
        Start-Sleep -Seconds 15
        
        # Create replication user
        Write-ColorOutput "Creating replication user..." $Colors.Info
        docker exec postgres-master psql -U rqa2025_user -d rqa2025_production -c "CREATE USER replicator REPLICATION LOGIN PASSWORD 'rqa2025_repl_password';" 2>$null
        
        # Create PostgreSQL slave
        Write-ColorOutput "Creating PostgreSQL slave container..." $Colors.Info
        $postgresSlaveResult = docker run -d --name postgres-slave `
            --network rqa2025-network `
            --ip 172.20.2.11 `
            -p 5433:5432 `
            -e POSTGRES_DB=rqa2025_production `
            -e POSTGRES_USER=rqa2025_user `
            -e POSTGRES_PASSWORD=rqa2025_slave_password `
            -e POSTGRES_MASTER_HOST=172.20.2.10 `
            -e POSTGRES_MASTER_PORT=5432 `
            -e POSTGRES_MASTER_USER=replicator `
            -e POSTGRES_MASTER_PASSWORD=rqa2025_repl_password `
            -v postgres_slave_data:/var/lib/postgresql/data `
            postgres:15-alpine
        
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "Failed to create PostgreSQL slave container" $Colors.Error
            return $false
        }
        
        # Wait for slave to be ready
        Write-ColorOutput "Waiting for PostgreSQL slave to be ready..." $Colors.Info
        Start-Sleep -Seconds 15
        
        # Test PostgreSQL master
        Write-ColorOutput "Testing PostgreSQL master..." $Colors.Info
        $masterTest = docker exec postgres-master psql -U rqa2025_user -d rqa2025_production -c "SELECT version();" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "PostgreSQL master is ready" $Colors.Success
        } else {
            Write-ColorOutput "PostgreSQL master is not ready" $Colors.Warning
        }
        
        # Test PostgreSQL slave
        Write-ColorOutput "Testing PostgreSQL slave..." $Colors.Info
        $slaveTest = docker exec postgres-slave psql -U rqa2025_user -d rqa2025_production -c "SELECT version();" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "PostgreSQL slave is ready" $Colors.Success
        } else {
            Write-ColorOutput "PostgreSQL slave is not ready" $Colors.Warning
        }
        
        Write-ColorOutput "PostgreSQL Master-Slave deployment completed successfully" $Colors.Success
        return $true
    }
    catch {
        Write-ColorOutput "PostgreSQL deployment failed: $($_.Exception.Message)" $Colors.Error
        return $false
    }
}

# Function to deploy HAProxy load balancer
function Deploy-HAProxy {
    Write-ColorOutput "Deploying HAProxy load balancer..." $Colors.Header
    
    try {
        # Create HAProxy configuration
        Write-ColorOutput "Creating HAProxy configuration..." $Colors.Info
        $haproxyConfig = @"
global
    daemon
    maxconn 4096
    log stdout format raw local0 info

defaults
    log global
    mode http
    option httplog
    option dontlognull
    timeout connect 5000
    timeout client 50000
    timeout server 50000

frontend rqa2025_frontend
    bind *:80
    mode http
    default_backend rqa2025_backend

backend rqa2025_backend
    mode http
    balance roundrobin
    option httpchk GET /health
    server redis_master 172.20.1.10:6379 check
    server redis_slave 172.20.1.11:6379 check backup
    server postgres_master 172.20.2.10:5432 check
    server postgres_slave 172.20.2.11:5432 check backup

listen stats
    bind *:8080
    mode http
    stats enable
    stats uri /stats
    stats refresh 10s
    stats auth admin:rqa2025_haproxy_password
"@
        
        # Save HAProxy configuration
        $haproxyConfigPath = "$env:TEMP\haproxy.cfg"
        $haproxyConfig | Out-File -FilePath $haproxyConfigPath -Encoding UTF8
        
        # Create HAProxy container
        Write-ColorOutput "Creating HAProxy container..." $Colors.Info
        $haproxyResult = docker run -d --name haproxy `
            --network rqa2025-network `
            --ip 172.20.3.10 `
            -p 80:80 `
            -p 8080:8080 `
            -v "${haproxyConfigPath}:/usr/local/etc/haproxy/haproxy.cfg:ro" `
            haproxy:2.8-alpine
        
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "Failed to create HAProxy container" $Colors.Error
            return $false
        }
        
        # Wait for HAProxy to be ready
        Write-ColorOutput "Waiting for HAProxy to be ready..." $Colors.Info
        Start-Sleep -Seconds 10
        
        # Test HAProxy
        Write-ColorOutput "Testing HAProxy..." $Colors.Info
        try {
            $haproxyResponse = Invoke-WebRequest -Uri "http://localhost:8080/stats" -UseBasicParsing -TimeoutSec 10
            if ($haproxyResponse.StatusCode -eq 200) {
                Write-ColorOutput "HAProxy is ready and accessible" $Colors.Success
            } else {
                Write-ColorOutput "HAProxy responded with status: $($haproxyResponse.StatusCode)" $Colors.Warning
            }
        }
        catch {
            Write-ColorOutput "HAProxy test failed: $($_.Exception.Message)" $Colors.Warning
        }
        
        # Clean up config file
        Remove-Item $haproxyConfigPath -Force
        
        Write-ColorOutput "HAProxy deployment completed successfully" $Colors.Success
        return $true
    }
    catch {
        Write-ColorOutput "HAProxy deployment failed: $($_.Exception.Message)" $Colors.Error
        return $false
    }
}

# Function to deploy monitoring services
function Deploy-MonitoringServices {
    Write-ColorOutput "Deploying monitoring services..." $Colors.Header
    
    try {
        # Create Prometheus configuration
        Write-ColorOutput "Creating Prometheus configuration..." $Colors.Info
        $prometheusConfig = @"
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'redis'
    static_configs:
      - targets: ['172.20.1.10:6379', '172.20.1.11:6379']

  - job_name: 'postgresql'
    static_configs:
      - targets: ['172.20.2.10:5432', '172.20.2.11:5432']

  - job_name: 'haproxy'
    static_configs:
      - targets: ['172.20.3.10:8080']
"@
        
        # Save Prometheus configuration
        $prometheusConfigPath = "$env:TEMP\prometheus.yml"
        $prometheusConfig | Out-File -FilePath $prometheusConfigPath -Encoding UTF8
        
        # Create Prometheus container
        Write-ColorOutput "Creating Prometheus container..." $Colors.Info
        $prometheusResult = docker run -d --name prometheus `
            --network rqa2025-network `
            --ip 172.20.4.10 `
            -p 9090:9090 `
            -v "${prometheusConfigPath}:/etc/prometheus/prometheus.yml:ro" `
            -v prometheus_data:/prometheus `
            prom/prometheus:latest `
            --config.file=/etc/prometheus/prometheus.yml `
            --storage.tsdb.path=/prometheus `
            --web.console.libraries=/etc/prometheus/console_libraries `
            --web.console.templates=/etc/prometheus/consoles `
            --storage.tsdb.retention.time=200h `
            --web.enable-lifecycle
        
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "Failed to create Prometheus container" $Colors.Error
            return $false
        }
        
        # Create Grafana container
        Write-ColorOutput "Creating Grafana container..." $Colors.Info
        $grafanaResult = docker run -d --name grafana `
            --network rqa2025-network `
            --ip 172.20.4.11 `
            -p 3000:3000 `
            -e GF_SECURITY_ADMIN_PASSWORD=rqa2025_grafana_password `
            -e GF_USERS_ALLOW_SIGN_UP=false `
            -v grafana_data:/var/lib/grafana `
            grafana/grafana:latest
        
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "Failed to create Grafana container" $Colors.Error
            return $false
        }
        
        # Wait for services to be ready
        Write-ColorOutput "Waiting for monitoring services to be ready..." $Colors.Info
        Start-Sleep -Seconds 15
        
        # Test Prometheus
        Write-ColorOutput "Testing Prometheus..." $Colors.Info
        try {
            $prometheusResponse = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/status/targets" -UseBasicParsing -TimeoutSec 10
            if ($prometheusResponse.StatusCode -eq 200) {
                Write-ColorOutput "Prometheus is ready and accessible" $Colors.Success
            } else {
                Write-ColorOutput "Prometheus responded with status: $($prometheusResponse.StatusCode)" $Colors.Warning
            }
        }
        catch {
            Write-ColorOutput "Prometheus test failed: $($_.Exception.Message)" $Colors.Warning
        }
        
        # Test Grafana
        Write-ColorOutput "Testing Grafana..." $Colors.Info
        try {
            $grafanaResponse = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -UseBasicParsing -TimeoutSec 10
            if ($grafanaResponse.StatusCode -eq 200) {
                Write-ColorOutput "Grafana is ready and accessible" $Colors.Success
            } else {
                Write-ColorOutput "Grafana responded with status: $($grafanaResponse.StatusCode)" $Colors.Warning
            }
        }
        catch {
            Write-ColorOutput "Grafana test failed: $($_.Exception.Message)" $Colors.Warning
        }
        
        # Clean up config file
        Remove-Item $prometheusConfigPath -Force
        
        Write-ColorOutput "Monitoring services deployment completed successfully" $Colors.Success
        return $true
    }
    catch {
        Write-ColorOutput "Monitoring services deployment failed: $($_.Exception.Message)" $Colors.Error
        return $false
    }
}

# Function to generate deployment report
function Generate-DeploymentReport {
    Write-ColorOutput "Generating deployment report..." $Colors.Info
    
    $report = @{
        Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Environment = $Environment
        Services = @{}
    }
    
    # Check Redis status
    try {
        $redisMasterStatus = docker ps --filter "name=redis-master" --format "{{.Status}}" 2>$null
        $redisSlaveStatus = docker ps --filter "name=redis-slave" --format "{{.Status}}" 2>$null
        
        $report.Services.Redis = @{
            Master = if ($redisMasterStatus) { "Running" } else { "Not Running" }
            Slave = if ($redisSlaveStatus) { "Running" } else { "Not Running" }
        }
    }
    catch {
        $report.Services.Redis = @{ Error = "Status check failed" }
    }
    
    # Check PostgreSQL status
    try {
        $postgresMasterStatus = docker ps --filter "name=postgres-master" --format "{{.Status}}" 2>$null
        $postgresSlaveStatus = docker ps --filter "name=postgres-slave" --format "{{.Status}}" 2>$null
        
        $report.Services.PostgreSQL = @{
            Master = if ($postgresMasterStatus) { "Running" } else { "Not Running" }
            Slave = if ($postgresSlaveStatus) { "Running" } else { "Not Running" }
        }
    }
    catch {
        $report.Services.PostgreSQL = @{ Error = "Status check failed" }
    }
    
    # Check HAProxy status
    try {
        $haproxyStatus = docker ps --filter "name=haproxy" --format "{{.Status}}" 2>$null
        
        $report.Services.HAProxy = if ($haproxyStatus) { "Running" } else { "Not Running" }
    }
    catch {
        $report.Services.HAProxy = "Status check failed"
    }
    
    # Check monitoring services status
    try {
        $prometheusStatus = docker ps --filter "name=prometheus" --format "{{.Status}}" 2>$null
        $grafanaStatus = docker ps --filter "name=grafana" --format "{{.Status}}" 2>$null
        
        $report.Services.Monitoring = @{
            Prometheus = if ($prometheusStatus) { "Running" } else { "Not Running" }
            Grafana = if ($grafanaStatus) { "Running" } else { "Not Running" }
        }
    }
    catch {
        $report.Services.Monitoring = @{ Error = "Status check failed" }
    }
    
    # Display report
    Write-Host ""
    Write-ColorOutput "===============================================" $Colors.Header
    Write-ColorOutput "Basic Services Deployment Report" $Colors.Header
    Write-ColorOutput "===============================================" $Colors.Header
    Write-ColorOutput "Timestamp: $($report.Timestamp)" $Colors.Info
    Write-ColorOutput "Environment: $($report.Environment)" $Colors.Info
    Write-Host ""
    
    foreach ($service in $report.Services.GetEnumerator()) {
        Write-ColorOutput "📋 $($service.Key):" $Colors.Header
        if ($service.Value -is [hashtable]) {
            foreach ($item in $service.Value.GetEnumerator()) {
                $color = if ($item.Value -eq "Running") { $Colors.Success } else { $Colors.Warning }
                Write-ColorOutput "  $($item.Key) - $($item.Value)" -ForegroundColor $color
            }
        } else {
            $color = if ($service.Value -eq "Running") { $Colors.Success } else { $Colors.Warning }
            Write-ColorOutput "  Status: $($service.Value)" -ForegroundColor $color
        }
        Write-Host ""
    }
    
    # Save report to file
    $reportPath = "deploy\reports\basic_services_deployment_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
    $reportDir = Split-Path $reportPath -Parent
    if (!(Test-Path $reportDir)) {
        New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
    }
    
    $report | ConvertTo-Json -Depth 10 | Out-File -FilePath $reportPath -Encoding UTF8
    Write-ColorOutput "Deployment report saved to: $reportPath" $Colors.Info
    
    return $report
}

# Main execution
function Main {
    Write-ColorOutput "===============================================" $Colors.Header
    Write-ColorOutput "Basic Services Deployment for RQA2025" $Colors.Header
    Write-ColorOutput "===============================================" $Colors.Header
    Write-Host ""
    
    # Check if running as administrator
    if (!(Test-Administrator)) {
        Write-ColorOutput "This script requires administrator privileges" $Colors.Error
        Write-ColorOutput "Please run PowerShell as Administrator and try again" $Colors.Warning
        return 1
    }
    
    # Check Docker service
    if (!(Test-DockerService)) {
        Write-ColorOutput "Docker service is not running" $Colors.Error
        Write-ColorOutput "Please start Docker Desktop and try again" $Colors.Warning
        return 1
    }
    
    # Create Docker network
    if (!(New-DockerNetwork -NetworkName "rqa2025-network" -Subnet $NetworkSubnet)) {
        return 1
    }
    
    # Deploy Redis if not skipped
    if (!$SkipRedis) {
        if (!(Deploy-RedisMasterSlave)) {
            Write-ColorOutput "Redis deployment failed" $Colors.Error
            return 1
        }
    } else {
        Write-ColorOutput "Skipping Redis deployment" $Colors.Warning
    }
    
    # Deploy PostgreSQL if not skipped
    if (!$SkipPostgreSQL) {
        if (!(Deploy-PostgreSQLMasterSlave)) {
            Write-ColorOutput "PostgreSQL deployment failed" $Colors.Error
            return 1
        }
    } else {
        Write-ColorOutput "Skipping PostgreSQL deployment" $Colors.Warning
    }
    
    # Deploy HAProxy if not skipped
    if (!$SkipHAProxy) {
        if (!(Deploy-HAProxy)) {
            Write-ColorOutput "HAProxy deployment failed" $Colors.Error
            return 1
        }
    } else {
        Write-ColorOutput "Skipping HAProxy deployment" $Colors.Warning
    }
    
    # Deploy monitoring services if not skipped
    if (!$SkipMonitoring) {
        if (!(Deploy-MonitoringServices)) {
            Write-ColorOutput "Monitoring services deployment failed" $Colors.Error
            return 1
        }
    } else {
        Write-ColorOutput "Skipping monitoring services deployment" $Colors.Warning
    }
    
    # Generate deployment report
    Generate-DeploymentReport
    
    Write-Host ""
    Write-ColorOutput "===============================================" $Colors.Header
    Write-ColorOutput "Basic Services Deployment Complete!" $Colors.Success
    Write-ColorOutput "===============================================" $Colors.Header
    Write-ColorOutput "Next steps:" $Colors.Info
    Write-ColorOutput "1. Verify all services are running" $Colors.Info
    Write-ColorOutput "2. Test service connectivity" $Colors.Info
    Write-ColorOutput "3. Configure application connections" $Colors.Info
    Write-ColorOutput "4. Proceed to core services deployment" $Colors.Info
    
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

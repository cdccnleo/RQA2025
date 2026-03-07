# Test Script for Phase 1 Infrastructure Scripts
# This script validates all PowerShell scripts and tests their functionality

param(
    [switch]$Verbose,
    [switch]$SkipExecution,
    [switch]$GenerateReport
)

# Set error action preference
$ErrorActionPreference = "Continue"

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

# Function to test script syntax
function Test-ScriptSyntax {
    param(
        [string]$ScriptPath,
        [string]$ScriptName
    )
    
    Write-ColorOutput "Testing syntax for: $ScriptName" $Colors.Info
    
    try {
        $syntaxResult = powershell -Command "Get-Command -Syntax '$ScriptPath' 2>&1"
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✓ $ScriptName syntax is valid" $Colors.Success
            return @{
                Name = $ScriptName
                Path = $ScriptPath
                SyntaxValid = $true
                SyntaxOutput = $syntaxResult
                Error = $null
            }
        } else {
            Write-ColorOutput "✗ $ScriptName syntax validation failed" $Colors.Error
            return @{
                Name = $ScriptName
                Path = $ScriptPath
                SyntaxValid = $false
                SyntaxOutput = $null
                Error = "Exit code: $LASTEXITCODE"
            }
        }
    }
    catch {
        Write-ColorOutput "✗ $ScriptName syntax test error: $($_.Exception.Message)" $Colors.Error
        return @{
            Name = $ScriptName
            Path = $ScriptPath
            SyntaxValid = $false
            SyntaxOutput = $null
            Error = $_.Exception.Message
        }
    }
}

# Function to test script parameters
function Test-ScriptParameters {
    param(
        [string]$ScriptPath,
        [string]$ScriptName
    )
    
    Write-ColorOutput "Testing parameters for: $ScriptName" $Colors.Info
    
    try {
        $helpResult = powershell -Command "Get-Help '$ScriptPath' -Parameter * 2>&1"
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✓ $ScriptName parameters are accessible" $Colors.Success
            return @{
                ParametersAccessible = $true
                HelpOutput = $helpResult
                Error = $null
            }
        } else {
            Write-ColorOutput "✗ $ScriptName parameter help failed" $Colors.Error
            return @{
                ParametersAccessible = $false
                HelpOutput = $null
                Error = "Exit code: $LASTEXITCODE"
            }
        }
    }
    catch {
        Write-ColorOutput "✗ $ScriptName parameter test error: $($_.Exception.Message)" $Colors.Error
        return @{
            ParametersAccessible = $false
            HelpOutput = $null
            Error = $_.Exception.Message
        }
    }
}

# Function to test script execution (dry run)
function Test-ScriptExecution {
    param(
        [string]$ScriptPath,
        [string]$ScriptName
    )
    
    if ($SkipExecution) {
        Write-ColorOutput "Skipping execution test for: $ScriptName" $Colors.Warning
        return @{
            ExecutionTested = $false
            DryRunSuccess = $null
            Error = "Skipped by user request"
        }
    }
    
    Write-ColorOutput "Testing execution (dry run) for: $ScriptName" $Colors.Info
    
    try {
        # Try to execute with -WhatIf if available, otherwise just validate
        $whatIfResult = powershell -Command "& '$ScriptPath' -WhatIf 2>&1"
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✓ $ScriptName dry run successful" $Colors.Success
            return @{
                ExecutionTested = $true
                DryRunSuccess = $true
                Error = $null
            }
        } else {
            # Try without -WhatIf, just check if script can be loaded
            $loadResult = powershell -Command "& '$ScriptPath' -GenerateReportOnly 2>&1"
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput "✓ $ScriptName can be loaded and executed" $Colors.Success
                return @{
                    ExecutionTested = $true
                    DryRunSuccess = $true
                    Error = $null
                }
            } else {
                Write-ColorOutput "✗ $ScriptName execution test failed" $Colors.Error
                return @{
                    ExecutionTested = $true
                    DryRunSuccess = $false
                    Error = "Exit code: $LASTEXITCODE"
                }
            }
        }
    }
    catch {
        Write-ColorOutput "✗ $ScriptName execution test error: $($_.Exception.Message)" $Colors.Error
        return @{
            ExecutionTested = $true
            DryRunSuccess = $false
            Error = $_.Exception.Message
        }
    }
}

# Function to generate test report
function Generate-TestReport {
    param(
        [array]$TestResults
    )
    
    Write-ColorOutput "Generating test report..." $Colors.Info
    
    $report = @{
        Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        TestType = "Phase 1 Scripts Validation"
        TotalScripts = $TestResults.Count
        PassedScripts = ($TestResults | Where-Object { $_.SyntaxValid -and $_.ParametersAccessible }).Count
        FailedScripts = ($TestResults | Where-Object { !$_.SyntaxValid -or !$_.ParametersAccessible }).Count
        Results = $TestResults
    }
    
    # Display summary
    Write-Host ""
    Write-ColorOutput "===============================================" $Colors.Header
    Write-ColorOutput "Phase 1 Scripts Test Report" $Colors.Header
    Write-ColorOutput "===============================================" $Colors.Header
    Write-ColorOutput "Timestamp: $($report.Timestamp)" $Colors.Info
    Write-ColorOutput "Total Scripts: $($report.TotalScripts)" $Colors.Info
    Write-ColorOutput "Passed: $($report.PassedScripts)" -ForegroundColor $Colors.Success
    Write-ColorOutput "Failed: $($report.FailedScripts)" -ForegroundColor $(if ($report.FailedScripts -gt 0) { $Colors.Error } else { $Colors.Success })
    Write-Host ""
    
    # Display detailed results
    foreach ($result in $TestResults) {
        Write-ColorOutput "📋 $($result.Name):" $Colors.Header
        
        # Syntax test result
        $syntaxColor = if ($result.SyntaxValid) { $Colors.Success } else { $Colors.Error }
        Write-ColorOutput "  Syntax: $(if ($result.SyntaxValid) { 'Valid' } else { 'Invalid' })" -ForegroundColor $syntaxColor
        
        # Parameters test result
        $paramColor = if ($result.ParametersAccessible) { $Colors.Success } else { $Colors.Error }
        Write-ColorOutput "  Parameters: $(if ($result.ParametersAccessible) { 'Accessible' } else { 'Not Accessible' })" -ForegroundColor $paramColor
        
        # Execution test result
        if ($result.ExecutionTested) {
            $execColor = if ($result.DryRunSuccess) { $Colors.Success } else { $Colors.Error }
            Write-ColorOutput "  Execution: $(if ($result.DryRunSuccess) { 'Success' } else { 'Failed' })" -ForegroundColor $execColor
        } else {
            Write-ColorOutput "  Execution: Skipped" $Colors.Warning
        }
        
        # Error details if any
        if ($result.Error) {
            Write-ColorOutput "  Error: $($result.Error)" $Colors.Error
        }
        
        Write-Host ""
    }
    
    # Save report to file
    if ($GenerateReport) {
        $reportPath = "deploy\reports\phase1_scripts_test_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
        $reportDir = Split-Path $reportPath -Parent
        if (!(Test-Path $reportDir)) {
            New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
        }
        
        $report | ConvertTo-Json -Depth 10 | Out-File -FilePath $reportPath -Encoding UTF8
        Write-ColorOutput "Test report saved to: $reportPath" $Colors.Info
    }
    
    return $report
}

# Main test execution
function Main {
    Write-ColorOutput "===============================================" $Colors.Header
    Write-ColorOutput "Phase 1 Infrastructure Scripts Test Suite" $Colors.Header
    Write-ColorOutput "===============================================" $Colors.Header
    Write-Host ""
    
    # Define scripts to test
    $scriptsToTest = @(
        @{
            Name = "prepare_server_environment.ps1"
            Path = "deploy\scripts\prepare_server_environment.ps1"
            Category = "Environment Preparation"
        },
        @{
            Name = "prepare_docker_environment.ps1"
            Path = "deploy\scripts\prepare_docker_environment.ps1"
            Category = "Environment Preparation"
        },
        @{
            Name = "deploy_basic_services.ps1"
            Path = "deploy\scripts\deploy_basic_services.ps1"
            Category = "Basic Services"
        },
        @{
            Name = "deploy_phase1_infrastructure.ps1"
            Path = "deploy\scripts\deploy_phase1_infrastructure.ps1"
            Category = "Main Orchestration"
        }
    )
    
    $testResults = @()
    
    # Test each script
    foreach ($script in $scriptsToTest) {
        Write-ColorOutput "Testing $($script.Category): $($script.Name)" $Colors.Header
        Write-Host ""
        
        # Check if script exists
        if (!(Test-Path $script.Path)) {
            Write-ColorOutput "✗ Script not found: $($script.Path)" $Colors.Error
            $testResults += @{
                Name = $script.Name
                Path = $script.Path
                Category = $script.Category
                SyntaxValid = $false
                ParametersAccessible = $false
                ExecutionTested = $false
                Error = "Script file not found"
            }
            continue
        }
        
        # Test syntax
        $syntaxResult = Test-ScriptSyntax -ScriptPath $script.Path -ScriptName $script.Name
        
        # Test parameters
        $paramResult = Test-ScriptParameters -ScriptPath $script.Path -ScriptName $script.Name
        
        # Test execution
        $execResult = Test-ScriptExecution -ScriptPath $script.Path -ScriptName $script.Name
        
        # Combine results
        $testResults += @{
            Name = $script.Name
            Path = $script.Path
            Category = $script.Category
            SyntaxValid = $syntaxResult.SyntaxValid
            ParametersAccessible = $paramResult.ParametersAccessible
            ExecutionTested = $execResult.ExecutionTested
            DryRunSuccess = $execResult.DryRunSuccess
            Error = if ($syntaxResult.Error) { $syntaxResult.Error } elseif ($paramResult.Error) { $paramResult.Error } elseif ($execResult.Error) { $execResult.Error } else { $null }
        }
        
        Write-Host ""
    }
    
    # Generate and display test report
    $report = Generate-TestReport -TestResults $testResults
    
    # Final summary
    Write-ColorOutput "===============================================" $Colors.Header
    Write-ColorOutput "Test Summary" $Colors.Header
    Write-ColorOutput "===============================================" $Colors.Header
    
    if ($report.FailedScripts -eq 0) {
        Write-ColorOutput "🎉 All Phase 1 scripts passed validation!" $Colors.Success
        Write-ColorOutput "Ready for production deployment" $Colors.Success
    } else {
        Write-ColorOutput "⚠️  Some scripts failed validation" $Colors.Warning
        Write-ColorOutput "Please review the errors above before proceeding" $Colors.Warning
    }
    
    Write-Host ""
    Write-ColorOutput "Next steps:" $Colors.Info
    Write-ColorOutput "1. Review test results" $Colors.Info
    Write-ColorOutput "2. Fix any validation errors" $Colors.Info
    Write-ColorOutput "3. Run deployment scripts" $Colors.Info
    
    return if ($report.FailedScripts -eq 0) { 0 } else { 1 }
}

# Run main function
try {
    $exitCode = Main
    exit $exitCode
}
catch {
    Write-ColorOutput "Test execution failed: $($_.Exception.Message)" $Colors.Error
    exit 1
}

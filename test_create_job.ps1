$body = @{
    model_type = "RandomForest"
    config = @{
        symbols = @("002837", "688702", "000987")
        epochs = 5
    }
} | ConvertTo-Json -Depth 3

Invoke-WebRequest -Uri 'http://localhost/api/v1/ml/training/jobs' -Method POST -Body $body -ContentType 'application/json'

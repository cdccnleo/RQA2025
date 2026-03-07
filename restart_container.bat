@echo off
echo 重启RQA应用容器...
docker stop rqa2025-rqa2025-app-1
timeout /t 2 /nobreak > nul
docker rm rqa2025-rqa2025-app-1
timeout /t 2 /nobreak > nul
docker-compose up -d rqa2025-app
echo 容器重启完成
pause
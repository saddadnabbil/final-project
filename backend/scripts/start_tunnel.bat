@echo off
REM Cloudflare Tunnel Startup Script
REM Run this to start tunnel manually (if not using Windows Service)

echo ========================================
echo   Cloudflare Tunnel Startup
echo ========================================
echo.

REM Check if cloudflared is installed
where cloudflared >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: cloudflared not found!
    echo Please install from: https://github.com/cloudflare/cloudflared/releases
    pause
    exit /b 1
)

echo Found cloudflared: OK
echo.

REM Get tunnel name from user or use default
set TUNNEL_NAME=skripsi-sunda-backend

echo Starting tunnel: %TUNNEL_NAME%
echo.
echo Tunnel will run in foreground (Ctrl+C to stop)
echo For background service, use: cloudflared service install
echo.

REM Run tunnel
cloudflared tunnel run %TUNNEL_NAME%

pause

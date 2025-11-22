# Quick script to check training status
Write-Host "=== Training Status Check ===" -ForegroundColor Cyan
Write-Host ""

if (Test-Path "training_fast.log") {
    Write-Host "Training log found!" -ForegroundColor Green
    Write-Host "Last 20 lines:" -ForegroundColor Yellow
    Write-Host ""
    Get-Content training_fast.log -Tail 20
    Write-Host ""
    Write-Host "---" -ForegroundColor Gray
} else {
    Write-Host "Training log not found yet. Training may still be initializing..." -ForegroundColor Yellow
}

if (Test-Path "runs/clip_hf_chess_fast") {
    Write-Host "`nOutput directory exists!" -ForegroundColor Green
    $files = Get-ChildItem "runs/clip_hf_chess_fast" -ErrorAction SilentlyContinue
    if ($files) {
        Write-Host "Files in output directory:" -ForegroundColor Yellow
        $files | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
    }
} else {
    Write-Host "`nOutput directory not created yet." -ForegroundColor Yellow
}

Write-Host "`nTo monitor in real-time, run:" -ForegroundColor Cyan
Write-Host "  Get-Content training_fast.log -Wait -Tail 10" -ForegroundColor White


@echo off
echo Starting Faster Whisper...
echo.

:: 入力パス（単一ファイル or フォルダ）
set "INPUT=%~1"

:: 仮想環境の python 実行
set "PYTHON=C:\Users\nunif\fasterwhisper\Scripts\python.exe"
set "SCRIPT=F:\Tools\Whisper/Whisper.py"

"%PYTHON%" "%SCRIPT%" "%INPUT%"

echo.
echo Finished.
pause

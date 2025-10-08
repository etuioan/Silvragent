@echo off
setlocal

REM Sicherstellen, dass wir im Ordner des Skripts sind
cd /d "%~dp0"

REM Lade Umgebungsvariablen aus .env, falls vorhanden
if exist .env (
  for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" set "%%A=%%B"
  )
)

REM Falls DEEPSEEK_API_KEY nicht gesetzt ist, Hinweis ausgeben
if "%DEEPSEEK_API_KEY%"=="" (
  echo DEEPSEEK_API_KEY ist nicht gesetzt. Bitte setze ihn in .env oder als Umgebungsvariable.
  pause
  exit /b 1
)

REM Starte das gebaute Release-Binary
if exist target\release\hekmatki.exe (
  echo Starte hekmatki...
  REM Neues Fenster: zuerst in Skriptordner wechseln, dann Binary starten und offen lassen
  start "hekmatki" cmd /k cd /d "%~dp0" ^&^& target\release\hekmatki.exe
) else (
  echo Release-Binary nicht gefunden. Baue zuerst mit: cargo build --release
  pause
)

endlocal


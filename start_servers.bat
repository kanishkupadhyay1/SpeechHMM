@echo off
setlocal

echo Starting SpeechHMM Backend and Frontend...
start "SpeechHMM Backend" cmd /k "title SpeechHMM Backend && python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000"

cd frontend
start "SpeechHMM Frontend" cmd /k "title SpeechHMM Frontend && npm run dev"
cd ..

timeout /t 2 >nul
start http://localhost:5173
echo Application launched at http://localhost:5173

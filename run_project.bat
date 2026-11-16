@echo off
setlocal enabledelayedexpansion

echo ===============================================================================
echo     AI-POWERED SPEECH COMMAND RECOGNITION USING HIDDEN MARKOV MODELS
echo ===============================================================================

:: 1. Detect Python
set PYTHON_CMD=python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.11+.
    pause
    exit /b 1
)
echo [OK] Python detected.

:: 2. Check / Setup Environment
if exist "venv\Scripts\python.exe" (
    echo [INFO] Using existing virtual environment...
    set PYTHON_CMD=venv\Scripts\python.exe
) else (
    echo [INFO] Checking dependencies in active Python environment...
    python -m pip install -r requirements.txt --quiet
)

:: 3. Dataset check & download
if not exist "data\raw\speech_commands_v0.02\validation_list.txt" (
    echo [INFO] Downloading official Google Speech Commands v0.02 dataset...
    %PYTHON_CMD% src\data\download_dataset.py
) else (
    echo [OK] Official Google Speech Commands v0.02 dataset already present.
)

:: 4. Validate dataset
echo [INFO] Validating dataset integrity and splits...
%PYTHON_CMD% src\data\validate_dataset.py
if %errorlevel% neq 0 (
    echo [ERROR] Dataset validation failed.
    pause
    exit /b 1
)

:: 5. Check / Generate Manifests
if not exist "data\processed\train_manifest.csv" (
    echo [INFO] Generating train, validation, and test manifests...
    %PYTHON_CMD% src\data\prepare_dataset.py
)

:: 6. Check / Extract MFCC features
if not exist "data\features\test_features.joblib" (
    echo [INFO] Extracting test set MFCC features...
    %PYTHON_CMD% -m src.features.mfcc_extractor --manifest data\processed\test_manifest.csv --output data\features\test_features.joblib
)
if not exist "data\features\train_features.joblib" (
    echo [INFO] Extracting training set MFCC features...
    %PYTHON_CMD% -m src.features.mfcc_extractor --manifest data\processed\train_manifest.csv --output data\features\train_features.joblib
)

:: 7. Check / Train HMM Models
if exist "models\hmm_models\yes.pkl" (
    echo [OK] Trained HMM models already exist.
) else (
    echo [INFO] Training command-specific HMMs via Baum-Welch EM...
    %PYTHON_CMD% -c "from src.models.hmm_trainer import CommandHMMTrainer; CommandHMMTrainer().train_all('data/features/train_features.joblib')"
)

:: 8. Evaluation & Visualizations
if not exist "results\metrics\evaluation_metrics.json" (
    echo [INFO] Evaluating HMM models on official held-out test split...
    %PYTHON_CMD% -m src.models.model_evaluation
    echo [INFO] Generating evaluation and acoustic plots...
    %PYTHON_CMD% -m src.visualization.generate_all_plots
) else (
    echo [OK] Evaluation metrics and plots already present.
)

:: 9. Start Frontend & Backend
echo ===============================================================================
echo Starting FastAPI Backend (http://localhost:8000)
echo Starting React Vite Frontend (http://localhost:5173)
echo ===============================================================================

start "SpeechHMM Backend" cmd /k "title SpeechHMM Backend && %PYTHON_CMD% -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000"

cd frontend
if not exist "node_modules" (
    echo [INFO] Installing frontend npm packages...
    call npm install
)
start "SpeechHMM Frontend" cmd /k "title SpeechHMM Frontend && npm run dev"
cd ..

echo Waiting for servers to initialize...
timeout /t 3 >nul
start http://localhost:5173
echo.
echo Application successfully launched!
echo - Web Dashboard: http://localhost:5173
echo - API Docs:      http://localhost:8000/docs
echo.
pause

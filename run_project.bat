@echo off
setlocal enabledelayedexpansion

echo ===============================================================================
echo     AI-POWERED SPEECH COMMAND RECOGNITION USING HIDDEN MARKOV MODELS
echo ===============================================================================

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.11+.
    pause
    exit /b 1
)
echo [OK] Python detected.

:: 2. Check Virtual Environment
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Creating Python virtual environment 'venv'...
    python -m venv venv
)
call venv\Scripts\activate.bat

:: 3. Install requirements
echo [INFO] Checking Python dependencies...
python -m pip install -r requirements.txt --quiet

:: 4. Dataset check & download
if not exist "data\raw\speech_commands_v0.02\validation_list.txt" (
    echo [INFO] Downloading official Google Speech Commands v0.02 dataset...
    python src\data\download_dataset.py
) else (
    echo [OK] Official Google Speech Commands v0.02 dataset already present.
)

:: 5. Validate dataset
echo [INFO] Validating dataset integrity and splits...
python src\data\validate_dataset.py
if %errorlevel% neq 0 (
    echo [ERROR] Dataset validation failed.
    pause
    exit /b 1
)

:: 6. Check / Generate Manifests
if not exist "data\processed\train_manifest.csv" (
    echo [INFO] Generating train, validation, and test manifests...
    python src\data\prepare_dataset.py
)

:: 7. Check / Extract MFCC features
if not exist "data\features\test_features.joblib" (
    echo [INFO] Extracting test set MFCC features...
    python -m src.features.mfcc_extractor --manifest data\processed\test_manifest.csv --output data\features\test_features.joblib
)
if not exist "data\features\train_features.joblib" (
    echo [INFO] Extracting training set MFCC features...
    python -m src.features.mfcc_extractor --manifest data\processed\train_manifest.csv --output data\features\train_features.joblib
)

:: 8. Check / Train HMM Models
if exist "models\hmm_models\yes.pkl" (
    echo [OK] Trained HMM models already exist.
    set /p RETRAIN="Do you want to retrain models? (y/N): "
    if /i "!RETRAIN!"=="y" (
        echo [INFO] Retraining command HMMs via Baum-Welch EM...
        python -c "from src.models.hmm_trainer import CommandHMMTrainer; CommandHMMTrainer().train_all('data/features/train_features.joblib')"
    )
) else (
    echo [INFO] Training command-specific HMMs via Baum-Welch EM...
    python -c "from src.models.hmm_trainer import CommandHMMTrainer; CommandHMMTrainer().train_all('data/features/train_features.joblib')"
)

:: 9. Evaluation
if not exist "results\metrics\evaluation_metrics.json" (
    echo [INFO] Evaluating HMM models on official held-out test split...
    python src\models\model_evaluation.py
    echo [INFO] Generating evaluation and acoustic plots...
    python src\visualization\evaluation_plots.py
) else (
    echo [OK] Evaluation metrics already generated.
)

:: 10. Start Frontend & Backend
echo ===============================================================================
echo Starting FastAPI Backend (http://localhost:8000)
echo Starting React Vite Frontend (http://localhost:5173)
echo ===============================================================================

start "SpeechHMM FastAPI Backend" cmd /k "venv\Scripts\activate.bat && uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"

cd frontend
if not exist "node_modules" (
    echo [INFO] Installing frontend npm packages...
    call npm install
)
start "SpeechHMM React Frontend" cmd /k "npm run dev"
cd ..

echo Application launched! Opening dashboard at http://localhost:5173...
timeout /t 3 >nul
start http://localhost:5173
pause

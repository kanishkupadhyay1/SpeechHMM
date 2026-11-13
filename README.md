# AI-Powered Speech Command Recognition Using Hidden Markov Models

A production-grade, mathematically grounded speech command recognition system using Continuous Gaussian Hidden Markov Models (HMM) trained on the official Google Speech Commands Dataset (v0.02).

---

## 1. Project Overview

### 1.1 Problem Statement
Spoken acoustic commands consist of time-varying, non-stationary acoustic observations. The challenge is recognizing an isolated limited vocabulary of 10 target spoken words across diverse speakers, accents, speaking rates, and ambient noise conditions without using deep neural networks or black-box classifiers.

### 1.2 Objectives
- Recognize limited-vocabulary spoken commands using Continuous Gaussian Hidden Markov Models.
- Extract 39-dimensional acoustic representations (13 MFCC + 13 Delta + 13 Delta-Delta) with pre-emphasis and Cepstral Mean Normalization (CMVN).
- Train command-specific Left-to-Right Bakis Gaussian HMMs using the Baum-Welch (Expectation-Maximization) algorithm.
- Implement an independent, numerically stable log-space **Forward Algorithm** for sequence scoring and likelihood comparison.
- Implement an independent log-space **Viterbi Algorithm** for optimal state sequence alignment and temporal decoding.
- Evaluate model accuracy, Macro F1, precision, recall, and confusion matrix on the official held-out test split.
- Provide a full-stack interactive web application with real browser microphone recording, live waveform/spectrogram visualization, and RESTful API endpoints.

---

## 2. Dataset Architecture

### 2.1 Official Source
- **Dataset:** Google Speech Commands Dataset v0.02
- **URL:** [https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz](https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
- **Archive Size:** ~2.42 GB (compressed), 105,878 total audio recordings
- **Audio Format:** 16,000 Hz, 16-bit PCM Mono WAV, duration ~1.0 second

### 2.2 Target Vocabulary (10 Core Commands)
`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`

### 2.3 Official Splits & Speaker Separation
Strict speaker separation is preserved according to the official `validation_list.txt` and `testing_list.txt` files to guarantee **zero speaker leakage** across splits:
- **Training Set:** 30,769 recordings
- **Validation Set:** 3,703 recordings
- **Held-Out Test Set:** 4,074 recordings
- **Total Validated Recordings:** 38,546 recordings

---

## 3. Acoustic Signal Processing & MFCC Extraction

Feature extraction transforms continuous acoustic pressure waves into compact, decorrelated acoustic representations:

```
Raw WAV (16 kHz)
   │
   ▼
DC Offset Removal & Pre-emphasis (alpha = 0.97)
   │
   ▼
Framing & Hamming Windowing (25ms window, 10ms hop)
   │
   ▼
Short-Time Fourier Transform (STFT, N_FFT = 512)
   │
   ▼
Mel-Filterbank Integration (40 triangular filters)
   │
   ▼
Logarithmic Compression & Discrete Cosine Transform (DCT-II)
   │
   ▼
13 Base MFCCs (c0 - c12)
   │
   ▼
First & Second Order Derivatives (13 Delta + 13 Delta-Delta)
   │
   ▼
Cepstral Mean & Variance Normalization (CMVN)
   │
   ▼
Output: T x 39 Acoustic Feature Sequence
```

---

## 4. Hidden Markov Model Formulation

### 4.1 Topology & Parameters
For each command $w \in \{\text{yes, no, up, down, left, right, on, off, stop, go}\}$, a dedicated Continuous Gaussian HMM $\lambda_w = (A, B, \pi)$ is constructed:
- **Hidden States:** $N = 5$ states arranged in a Left-to-Right (Bakis) topology without backward transitions to reflect the temporal phone progression.
- **Emission Distribution:** Continuous Multivariate Gaussian with diagonal covariance matrix per state:
  $$b_j(x_t) = \mathcal{N}(x_t; \mu_j, \Sigma_j) = \frac{1}{(2\pi)^{D/2} |\Sigma_j|^{1/2}} \exp\left( -\frac{1}{2} (x_t - \mu_j)^T \Sigma_j^{-1} (x_t - \mu_j) \right)$$
- **Initial State Distribution:** $\pi = [1.0, 0.0, 0.0, 0.0, 0.0]^T$.

### 4.2 Baum-Welch EM Training
Parameters $\lambda_w$ are iteratively optimized to maximize the total training log likelihood:
$$\max_{\lambda_w} \sum_{k=1}^{K_w} \ln P(X^{(k)} \mid \lambda_w)$$
- **E-Step:** Compute forward ($\alpha_t$) and backward ($\beta_t$) variables; compute posterior state occupation probabilities $\gamma_t(i)$ and transition probabilities $\xi_t(i, j)$.
- **M-Step:** Re-estimate $\pi_i$, $a_{ij}$, $\mu_j$, and diagonal variances $\sigma_{j,d}^2$.

---

## 5. Forward & Viterbi Algorithms

### 5.1 Log-Space Forward Algorithm
Given an observation sequence $X = (x_1, \dots, x_T)$, the forward trellis variable $\ln \alpha_t(j)$ is computed strictly in natural log space:
1. **Initialization ($t = 1$):**
   $$\ln \alpha_1(i) = \ln \pi_i + \ln b_i(x_1)$$
2. **Induction ($t = 2 \dots T$):**
   $$\ln \alpha_t(j) = \text{logsumexp}_{i=1}^N \left( \ln \alpha_{t-1}(i) + \ln a_{ij} \right) + \ln b_j(x_t)$$
3. **Termination:**
   $$\ln P(X \mid \lambda_w) = \text{logsumexp}_{i=1}^N \left( \ln \alpha_T(i) \right)$$

### 5.2 Command Classification
Classification selects the model with maximum forward log-likelihood:
$$\hat{w} = \arg\max_{w \in \mathcal{V}} \ln P(X \mid \lambda_w)$$

### 5.3 Viterbi Decoding
Finds the single best state alignment path $Q^* = (q_1^*, \dots, q_T^*)$:
1. **Recursion:**
   $$\ln \delta_t(j) = \max_{i=1 \dots N} \left[ \ln \delta_{t-1}(i) + \ln a_{ij} \right] + \ln b_j(x_t)$$
   $$\psi_t(j) = \arg\max_{i=1 \dots N} \left[ \ln \delta_{t-1}(i) + \ln a_{ij} \right]$$
2. **Backtracking:**
   $$q_T^* = \arg\max_{i} \ln \delta_T(i), \quad q_t^* = \psi_{t+1}(q_{t+1}^*)$$

---

## 6. Official Test Set Evaluation Results

Evaluated across all **4,074 held-out recordings** from the official Google Speech Commands v0.02 test split (zero speaker overlap):

| Metric | Empirical Result | Notes |
|---|---|---|
| **Test Accuracy** | **55.87%** | Random baseline is 10.0% for 10 classes |
| **Macro Precision** | **56.29%** | Unweighted mean across all 10 classes |
| **Macro Recall** | **55.61%** | Unweighted mean across all 10 classes |
| **Macro F1-Score** | **54.65%** | Harmonic mean of precision & recall |
| **Weighted F1-Score**| **54.80%** | Weighted by class support counts |
| **Average Latency** | **3.61 ms** | Full 10-HMM forward pass per 1-second sample |

### Per-Command Breakdown

| Command | Support (Samples) | Precision (%) | Recall (%) | F1-Score (%) |
|---|---|---|---|---|
| **yes** | 419 | **82.81%** | **81.62%** | **82.21%** |
| **stop** | 411 | **77.22%** | **86.62%** | **81.65%** |
| **right** | 396 | **67.25%** | **58.08%** | **62.33%** |
| **off** | 402 | **56.20%** | **57.46%** | **56.83%** |
| **no** | 405 | **47.61%** | **66.42%** | **55.46%** |
| **up** | 425 | **42.90%** | **68.24%** | **52.68%** |
| **left** | 412 | **48.16%** | **50.73%** | **49.41%** |
| **go** | 402 | **42.41%** | **34.08%** | **37.79%** |
| **on** | 396 | **58.66%** | **26.52%** | **36.52%** |
| **down** | 406 | **39.63%** | **26.35%** | **31.66%** |

*All metrics are verified from actual evaluations stored in `results/metrics/evaluation_metrics.json` and `results/predictions/test_predictions.csv`.*

---

## 6. Directory Structure

```
speech-command-recognition-hmm/
│
├── data/
│   ├── raw/                       # Downloaded archive and extracted WAVs
│   ├── processed/                 # Train, validation, test manifest CSVs
│   └── features/                  # Precomputed 39D MFCC feature caches
│
├── models/
│   ├── hmm_models/                # Serialized command HMMs (yes.pkl, ...)
│   ├── label_encoder.pkl          # Command vocabulary mapping
│   └── model_config.json          # Hyperparameter specification
│
├── results/
│   ├── metrics/                   # evaluation_metrics.json, classification_report.csv
│   ├── predictions/               # test_predictions.csv
│   └── plots/                     # Confusion matrix, accuracy, spectrograms
│
├── src/
│   ├── data/                      # download_dataset.py, validate_dataset.py, prepare_dataset.py
│   ├── features/                  # mfcc_extractor.py
│   ├── models/                    # hmm_trainer.py, forward_algorithm.py, viterbi_decoder.py, model_evaluation.py
│   ├── visualization/             # mfcc_visualization.py, evaluation_plots.py, likelihood_visualization.py
│   └── api/                       # main.py, schemas.py, prediction_service.py
│
├── frontend/                      # React 18 + Vite + Tailwind + Recharts
│   ├── src/
│   │   ├── components/            # MicrophoneRecorder, ConfusionMatrix, MFCCChart, ...
│   │   ├── pages/                 # Dashboard, LiveRecognition, ModelAnalysis, About
│   │   └── services/api.js        # Axios REST client
│   └── package.json
│
├── tests/                         # Pytest automated test suite
├── requirements.txt
├── run_project.bat
└── README.md
```

---

## 7. Installation & Quick Start

### 7.1 Automated One-Click Launcher (Windows)
Double-click or run:
```cmd
run_project.bat
```
This script checks Python, creates `venv`, installs dependencies, downloads Google Speech Commands v0.02, validates audio, extracts MFCCs, trains HMMs, evaluates on the test split, generates plots, and starts both backend and frontend.

### 7.2 Manual Step-by-Step Execution
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and validate dataset
python src/data/download_dataset.py
python src/data/validate_dataset.py

# 3. Build manifest CSVs
python src/data/prepare_dataset.py

# 4. Extract MFCC features
python -m src.features.mfcc_extractor --manifest data/processed/test_manifest.csv --output data/features/test_features.joblib
python -m src.features.mfcc_extractor --manifest data/processed/train_manifest.csv --output data/features/train_features.joblib

# 5. Train HMMs
python -c "from src.models.hmm_trainer import CommandHMMTrainer; CommandHMMTrainer().train_all('data/features/train_features.joblib')"

# 6. Evaluate on held-out test split
python src/models/model_evaluation.py
python src/visualization/evaluation_plots.py

# 7. Start FastAPI Backend
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 8. Start React Frontend
cd frontend
npm install
npm run dev
```

---

## 8. Web Dashboard & Microphone Interface
- **Dashboard:** `http://localhost:5173`
- **FastAPI Interactive Docs:** `http://localhost:8000/docs`
- **Live Recognition:** Uses browser `navigator.mediaDevices.getUserMedia` via `MediaRecorder` to capture real microphone voice, converted in-browser to 16 kHz Mono WAV, processed on the backend via trained Gaussian HMMs.

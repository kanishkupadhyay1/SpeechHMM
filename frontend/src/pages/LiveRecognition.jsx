import React, { useState } from 'react';
import MicrophoneRecorder from '../components/MicrophoneRecorder';
import RecognitionResult from '../components/RecognitionResult';
import CommandLikelihoodChart from '../components/CommandLikelihoodChart';
import MFCCChart from '../components/MFCCChart';
import { recognizeAudio, getMFCCData, getApiBase } from '../services/api';
import { Sparkles, HelpCircle, Loader2 } from 'lucide-react';

export default function LiveRecognition() {
  const [result, setResult] = useState(null);
  const [mfccData, setMfccData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAudioRecorded = async (audioBlob) => {
    try {
      setLoading(true);
      setError(null);

      // Perform parallel recognition and MFCC extraction on the genuine audio recording
      const [recResult, mfccResult] = await Promise.all([
        recognizeAudio(audioBlob),
        getMFCCData(audioBlob),
      ]);

      setResult(recResult);
      setMfccData(mfccResult);
    } catch (err) {
      console.error('Recognition error:', err);
      let detail = err.response?.data?.detail;
      if (!detail) {
        if (err.message === 'Network Error' || err.code === 'ERR_NETWORK') {
          const target = getApiBase() || window.location.origin;
          detail = `Unable to connect to backend server at ${target}. Please verify that the backend is running and accessible from your browser.`;
        } else {
          detail = err.message || 'Recognition request failed.';
        }
      }
      setError(`Backend error: ${detail}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Live Acoustic Recognition</h1>
        <p className="text-sm text-slate-500 mt-1">
          Capture user speech directly through browser microphone APIs and decode in real-time via trained Hidden Markov Models.
        </p>
      </div>

      {/* Recording Control */}
      <MicrophoneRecorder onAudioRecorded={handleAudioRecorded} isLoading={loading} />

      {/* Loading state indicator */}
      {loading && (
        <div className="p-8 rounded-2xl bg-white border border-indigo-100 shadow-sm flex flex-col items-center justify-center space-y-3">
          <Loader2 className="w-8 h-8 text-indigo-600 animate-spin" />
          <p className="text-sm font-semibold text-slate-700">
            Preprocessing audio & evaluating Forward Trellis across 10 command HMMs...
          </p>
          <span className="text-xs text-slate-400 font-mono">
            16 kHz mono → 13 MFCC + Δ + ΔΔ → Baum-Welch Forward Scoring → Viterbi Path
          </span>
        </div>
      )}

      {/* Error Banner */}
      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* Real Recognition Output */}
      {result && !loading && (
        <div className="space-y-6">
          <RecognitionResult result={result} />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <CommandLikelihoodChart
              scores={result.scores}
              predictedCommand={result.predicted_command}
            />
            <MFCCChart mfccData={mfccData} />
          </div>
        </div>
      )}

      {/* Guidelines & Tips */}
      {!result && !loading && (
        <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-sm">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2 mb-3">
            <HelpCircle className="w-4 h-4 text-indigo-600" />
            Tips for High-Accuracy Recognition
          </h3>
          <ul className="text-xs text-slate-600 space-y-2 list-disc list-inside">
            <li>
              Speak one of the 10 supported words clearly: <strong className="text-slate-800">yes, no, up, down, left, right, on, off, stop, go</strong>.
            </li>
            <li>
              Keep your utterance concise (~0.5s to 1.0s). The model is designed for 1-second isolated spoken commands.
            </li>
            <li>
              Minimize background noise or microphone clipping. Pre-emphasis and Cepstral Mean Normalization (CMVN) filter acoustic channel drift.
            </li>
            <li>
              Recognition uses mathematical Forward log-likelihood scoring, not Web Speech API, directly from Python GaussianHMM parameters.
            </li>
          </ul>
        </div>
      )}
    </div>
  );
}

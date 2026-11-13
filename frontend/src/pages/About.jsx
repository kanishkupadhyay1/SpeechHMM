import React from 'react';
import { BookOpen, Layers, Zap, CheckCircle2, ShieldCheck, Terminal } from 'lucide-react';

export default function About() {
  return (
    <div className="space-y-10 max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div>
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Academic & Architectural Overview</h1>
        <p className="text-base text-slate-600 mt-2 leading-relaxed">
          Theoretical foundations and algorithmic implementation details for acoustic speech command recognition using Continuous Gaussian Hidden Markov Models.
        </p>
      </div>

      {/* Problem Statement */}
      <section className="bg-white p-7 rounded-2xl border border-slate-200/80 shadow-sm space-y-3">
        <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-indigo-600" />
          1. Problem Statement
        </h2>
        <p className="text-sm text-slate-600 leading-relaxed">
          Spoken acoustic commands consist of time-varying non-stationary acoustic observations. The challenge is recognizing an isolated limited vocabulary of 10 target speech commands (<strong>yes, no, up, down, left, right, on, off, stop, go</strong>) across varied speakers, accents, speaking rates, and background noise conditions using probabilistic generative models rather than black-box deep neural networks.
        </p>
      </section>

      {/* Dataset Description */}
      <section className="bg-white p-7 rounded-2xl border border-slate-200/80 shadow-sm space-y-3">
        <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
          <Layers className="w-5 h-5 text-purple-600" />
          2. Google Speech Commands Dataset (v0.02)
        </h2>
        <p className="text-sm text-slate-600 leading-relaxed">
          The project utilizes the official <strong>Google Speech Commands Dataset v0.02</strong> archive, comprising over 105,000 one-second audio recordings at 16,000 Hz. The dataset includes explicit <code>validation_list.txt</code> and <code>testing_list.txt</code> splits designed by Google researchers to ensure strict speaker separation, preventing speaker identity leakage between train, validation, and test sets.
        </p>
      </section>

      {/* MFCC Acoustic Features */}
      <section className="bg-white p-7 rounded-2xl border border-slate-200/80 shadow-sm space-y-3">
        <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
          <Zap className="w-5 h-5 text-amber-500" />
          3. Mel-Frequency Cepstral Coefficients (MFCC) Extraction
        </h2>
        <div className="text-sm text-slate-600 leading-relaxed space-y-2">
          <p>
            Feature extraction transforms 1D time-domain waveforms into informative acoustic representations:
          </p>
          <ul className="list-disc list-inside space-y-1.5 ml-2 text-slate-700">
            <li><strong>Pre-emphasis:</strong> High-pass filter y[t] = x[t] - 0.97 x[t-1] compensates for the spectral roll-off of the human vocal tract.</li>
            <li><strong>Framing & Windowing:</strong> 25ms Hamming window (400 samples) with 10ms frame shift (160 samples hop length).</li>
            <li><strong>Mel Filter Bank:</strong> Tri-angular bandpass filters mapped according to the non-linear human auditory pitch perception scale.</li>
            <li><strong>DCT & Cepstrum:</strong> 13 base cepstral coefficients (c0 - c12) capturing vocal tract spectral shape.</li>
            <li><strong>Dynamic Features:</strong> 13 first-order deltas (velocity) + 13 second-order delta-deltas (acceleration), producing a 39-dimensional acoustic feature vector per frame.</li>
            <li><strong>CMVN:</strong> Cepstral Mean and Variance Normalization across frames removes linear channel distortion.</li>
          </ul>
        </div>
      </section>

      {/* Forward and Viterbi Algorithms */}
      <section className="bg-white p-7 rounded-2xl border border-slate-200/80 shadow-sm space-y-4">
        <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-emerald-600" />
          4. HMM Algorithms: Forward & Viterbi Decoding
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
            <h3 className="font-bold text-sm text-slate-900">Forward Algorithm (Log-Space Likelihood)</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Computes the total probability ln P(X | λ_k) of the acoustic observation sequence X under each command model λ_k. Numerically stable <code>logsumexp</code> prevents underflow across the 100-frame sequence. The command with maximum forward likelihood is selected as the predicted label.
            </p>
          </div>
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
            <h3 className="font-bold text-sm text-slate-900">Viterbi Algorithm (Optimal State Sequence)</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Finds the single most probable sequence of hidden states Q* that generated the observed speech frames. Using dynamic programming trellis search and backpointers, Viterbi reveals temporal phone/state progression through word duration.
            </p>
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="bg-white p-7 rounded-2xl border border-slate-200/80 shadow-sm space-y-3">
        <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
          <Terminal className="w-5 h-5 text-indigo-600" />
          5. Technology Stack
        </h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
            <div className="font-bold text-slate-900">hmmlearn</div>
            <div className="text-slate-500">Gaussian HMM & Baum-Welch</div>
          </div>
          <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
            <div className="font-bold text-slate-900">Librosa & SoundFile</div>
            <div className="text-slate-500">Acoustic Signal Processing</div>
          </div>
          <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
            <div className="font-bold text-slate-900">FastAPI & Uvicorn</div>
            <div className="text-slate-500">Asynchronous REST API</div>
          </div>
          <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
            <div className="font-bold text-slate-900">React & Vite</div>
            <div className="text-slate-500">Academic Web Dashboard</div>
          </div>
        </div>
      </section>
    </div>
  );
}

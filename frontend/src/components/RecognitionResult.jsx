import React from 'react';
import { Award, Clock, Activity, Cpu } from 'lucide-react';

export default function RecognitionResult({ result }) {
  if (!result) return null;

  const {
    predicted_command,
    confidence_normalized,
    best_log_likelihood,
    processing_time_ms,
    viterbi_state_path,
  } = result;

  return (
    <div className="bg-gradient-to-br from-white to-slate-50 rounded-2xl border border-indigo-100 p-6 shadow-md shadow-indigo-50/50">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 pb-6 border-b border-slate-100">
        <div className="flex items-center space-x-4">
          <div className="w-16 h-16 rounded-2xl bg-indigo-600 flex items-center justify-center text-white text-3xl font-extrabold shadow-lg shadow-indigo-200">
            {predicted_command.toUpperCase()[0]}
          </div>
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-indigo-600 bg-indigo-50 px-2.5 py-1 rounded-full border border-indigo-100">
              Recognized Acoustic Command
            </span>
            <h2 className="text-3xl font-black text-slate-900 capitalize mt-1 tracking-tight">
              "{predicted_command}"
            </h2>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="bg-white px-4 py-3 rounded-xl border border-slate-200/70 shadow-sm">
            <div className="text-xs text-slate-500 font-medium">Confidence</div>
            <div className="text-lg font-bold text-emerald-600 mt-0.5">
              {(confidence_normalized * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-white px-4 py-3 rounded-xl border border-slate-200/70 shadow-sm">
            <div className="text-xs text-slate-500 font-medium">Log Likelihood</div>
            <div className="text-lg font-bold text-slate-800 mt-0.5">
              {best_log_likelihood.toFixed(1)}
            </div>
          </div>
          <div className="bg-white px-4 py-3 rounded-xl border border-slate-200/70 shadow-sm">
            <div className="text-xs text-slate-500 font-medium">Latency</div>
            <div className="text-lg font-bold text-indigo-600 mt-0.5">
              {processing_time_ms.toFixed(0)} ms
            </div>
          </div>
        </div>
      </div>

      {viterbi_state_path && viterbi_state_path.length > 0 && (
        <div className="mt-5">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-2">
            <span className="flex items-center gap-1.5">
              <Cpu className="w-4 h-4 text-purple-600" />
              Optimal Hidden State Sequence (Viterbi Trellis Path):
            </span>
            <span className="text-slate-400 font-mono">{viterbi_state_path.length} acoustic frames</span>
          </div>

          <div className="flex flex-wrap gap-1 bg-white p-3 rounded-xl border border-slate-200/80 max-h-24 overflow-y-auto">
            {viterbi_state_path.map((state, idx) => (
              <span
                key={idx}
                className={`text-[10px] font-mono px-2 py-0.5 rounded font-semibold ${
                  state === 0
                    ? 'bg-blue-100 text-blue-800'
                    : state === 1
                    ? 'bg-indigo-100 text-indigo-800'
                    : state === 2
                    ? 'bg-purple-100 text-purple-800'
                    : state === 3
                    ? 'bg-pink-100 text-pink-800'
                    : 'bg-emerald-100 text-emerald-800'
                }`}
                title={`Frame ${idx}: State ${state}`}
              >
                S{state}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

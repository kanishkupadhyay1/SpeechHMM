import React, { useState, useEffect } from 'react';
import { Target, Award, Clock, Database, Layers, ArrowUpRight, Mic, CheckCircle2 } from 'lucide-react';
import MetricCard from '../components/MetricCard';
import ConfusionMatrix from '../components/ConfusionMatrix';
import ModelEvaluationChart from '../components/ModelEvaluationChart';
import { getMetrics, getConfusionMatrix } from '../services/api';

export default function Dashboard({ setActivePage }) {
  const [metrics, setMetrics] = useState(null);
  const [cmData, setCmData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [m, cm] = await Promise.all([getMetrics(), getConfusionMatrix()]);
        setMetrics(m);
        setCmData(cm);
        setError(null);
      } catch (err) {
        console.error('Error fetching dashboard metrics:', err);
        setError('Metrics not yet available. Please complete dataset training and evaluation.');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const commands = [
    'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'
  ];

  return (
    <div className="space-y-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Hero Banner */}
      <div className="bg-gradient-to-r from-indigo-700 via-indigo-600 to-purple-600 rounded-3xl p-8 sm:p-10 text-white shadow-xl shadow-indigo-100 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div className="max-w-2xl space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 backdrop-blur-md border border-white/20 text-xs font-semibold tracking-wide">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            Statistical Acoustic Speech Processing
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight">
            AI-Powered Speech Command Recognition
          </h1>
          <p className="text-indigo-100 text-sm sm:text-base leading-relaxed">
            Continuous Gaussian Hidden Markov Models (HMM) trained on the official Google Speech Commands v0.02 dataset. Features 13-dimensional MFCC acoustic vectors with Δ and ΔΔ dynamic acceleration.
          </p>
        </div>

        <button
          onClick={() => setActivePage('live')}
          className="flex items-center space-x-2 bg-white text-indigo-700 hover:bg-indigo-50 font-bold px-6 py-3.5 rounded-2xl shadow-lg transition-all hover:scale-105 shrink-0"
        >
          <Mic className="w-5 h-5 text-indigo-600" />
          <span>Test Live Microphone</span>
          <ArrowUpRight className="w-4 h-4" />
        </button>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <MetricCard
          title="Held-Out Accuracy"
          value={metrics ? `${(metrics.accuracy * 100).toFixed(1)}%` : '--%'}
          subtitle={metrics ? `Tested on ${metrics.num_test_samples} real recordings` : 'Awaiting evaluation'}
          icon={Target}
          color="indigo"
        />
        <MetricCard
          title="Macro F1-Score"
          value={metrics ? `${(metrics.macro_f1 * 100).toFixed(1)}%` : '--%'}
          subtitle="Unweighted mean across all 10 classes"
          icon={Award}
          color="purple"
        />
        <MetricCard
          title="Vocabulary Commands"
          value={commands.length.toString()}
          subtitle="yes, no, up, down, left, right, on, off, stop, go"
          icon={Layers}
          color="emerald"
        />
        <MetricCard
          title="Average Inference"
          value={metrics ? `${(metrics.avg_inference_time_sec * 1000).toFixed(1)} ms` : '-- ms'}
          subtitle="Forward Trellis log-likelihood pass"
          icon={Clock}
          color="blue"
        />
      </div>

      {/* Supported Commands Badge Bar */}
      <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-sm flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="text-sm font-bold text-slate-900">Supported Spoken Vocabulary:</div>
        <div className="flex flex-wrap gap-2">
          {commands.map((cmd) => (
            <span
              key={cmd}
              className="px-3 py-1 rounded-lg bg-slate-100 border border-slate-200 text-xs font-semibold text-slate-700 capitalize flex items-center gap-1.5"
            >
              <CheckCircle2 className="w-3.5 h-3.5 text-indigo-600" />
              {cmd}
            </span>
          ))}
        </div>
      </div>

      {/* Main Charts & Matrix */}
      {metrics && cmData ? (
        <div className="space-y-8">
          <ModelEvaluationChart perClassMetrics={metrics.per_class_metrics} />
          <ConfusionMatrix
            matrix={cmData.matrix}
            commands={cmData.commands}
            accuracy={cmData.accuracy}
            macroF1={cmData.macro_f1}
          />
        </div>
      ) : loading ? (
        <div className="bg-white p-12 rounded-2xl border border-slate-200 text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-indigo-600 border-t-transparent"></div>
          <p className="mt-3 text-sm text-slate-500 font-medium">Loading evaluation results from backend...</p>
        </div>
      ) : (
        <div className="bg-white p-8 rounded-2xl border border-amber-200 text-center text-amber-800">
          <p className="font-semibold">{error}</p>
          <p className="text-xs text-amber-600 mt-1">
            Backend will populate charts as soon as training and evaluation complete.
          </p>
        </div>
      )}
    </div>
  );
}

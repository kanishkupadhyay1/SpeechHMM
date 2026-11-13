import React, { useState, useEffect } from 'react';
import { getMetrics, getConfusionMatrix, getModelAnalysis } from '../services/api';
import ConfusionMatrix from '../components/ConfusionMatrix';
import ModelEvaluationChart from '../components/ModelEvaluationChart';
import MetricCard from '../components/MetricCard';
import { Cpu, ShieldCheck, Target, Award, ListFilter } from 'lucide-react';

export default function ModelAnalysis() {
  const [metrics, setMetrics] = useState(null);
  const [cmData, setCmData] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const [m, cm, a] = await Promise.all([
          getMetrics(),
          getConfusionMatrix(),
          getModelAnalysis(),
        ]);
        setMetrics(m);
        setCmData(cm);
        setAnalysis(a);
      } catch (err) {
        console.error('Error fetching model analysis:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-16 text-center">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-indigo-600 border-t-transparent"></div>
        <p className="mt-3 text-sm text-slate-500 font-medium">Loading statistical analysis...</p>
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Hidden Markov Model Statistical Analysis</h1>
        <p className="text-sm text-slate-500 mt-1">
          Detailed breakdown of Baum-Welch training convergence, HMM topology, and held-out test split performance.
        </p>
      </div>

      {metrics && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          <MetricCard
            title="Overall Accuracy"
            value={`${(metrics.accuracy * 100).toFixed(1)}%`}
            subtitle="Held-out official test split"
            icon={Target}
            color="indigo"
          />
          <MetricCard
            title="Macro F1 Score"
            value={`${(metrics.macro_f1 * 100).toFixed(1)}%`}
            subtitle="Harmonic mean of precision & recall"
            icon={Award}
            color="purple"
          />
          <MetricCard
            title="Hidden States (N)"
            value={analysis?.n_states?.toString() || '5'}
            subtitle="Left-to-Right Bakis topology"
            icon={Cpu}
            color="emerald"
          />
          <MetricCard
            title="Feature Dimension (D)"
            value={analysis?.feature_dimensions?.toString() || '39'}
            subtitle="13 MFCC + 13 Δ + 13 ΔΔ"
            icon={ShieldCheck}
            color="blue"
          />
        </div>
      )}

      {/* Model Parameter Specifications */}
      <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-sm">
        <h3 className="text-base font-bold text-slate-900 mb-4">Acoustic Model Parameters</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
            <span className="font-bold text-slate-700 block mb-1">State Topology</span>
            <p className="text-slate-600">Left-to-Right Bakis architecture. Models phoneme sequence progression from word onset to offset.</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
            <span className="font-bold text-slate-700 block mb-1">Emission Distributions</span>
            <p className="text-slate-600">Continuous Multivariate Gaussian with diagonal covariance matrix (covars_diag) per state.</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
            <span className="font-bold text-slate-700 block mb-1">Optimization Algorithm</span>
            <p className="text-slate-600">Baum-Welch Expectation-Maximization (EM) fitted over concatenated command observation sequences.</p>
          </div>
        </div>
      </div>

      {metrics && (
        <div className="space-y-8">
          <ModelEvaluationChart perClassMetrics={metrics.per_class_metrics} />
          {cmData && (
            <ConfusionMatrix
              matrix={cmData.matrix}
              commands={cmData.commands}
              accuracy={cmData.accuracy}
              macroF1={cmData.macro_f1}
            />
          )}
        </div>
      )}
    </div>
  );
}

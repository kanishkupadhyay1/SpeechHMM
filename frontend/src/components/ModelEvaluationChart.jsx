import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

export default function ModelEvaluationChart({ perClassMetrics }) {
  if (!perClassMetrics) return null;

  const data = Object.entries(perClassMetrics).map(([cmd, m]) => ({
    command: cmd,
    precision: Number((m.precision * 100).toFixed(1)),
    recall: Number((m.recall * 100).toFixed(1)),
    f1_score: Number((m.f1_score * 100).toFixed(1)),
  }));

  return (
    <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-base font-bold text-slate-900">Per-Command Acoustic Performance</h3>
          <p className="text-xs text-slate-500">
            Precision, Recall, and F1-Score across the 10 spoken command categories
          </p>
        </div>
      </div>

      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
            <XAxis dataKey="command" tick={{ fontSize: 12, fontWeight: 500 }} stroke="#64748b" />
            <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} stroke="#94a3b8" />
            <Tooltip
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-slate-900 text-white p-3 rounded-xl text-xs shadow-lg space-y-1">
                      <div className="font-bold capitalize text-sm">{label}</div>
                      <div className="text-blue-400">Precision: {payload[0].value}%</div>
                      <div className="text-purple-400">Recall: {payload[1].value}%</div>
                      <div className="text-emerald-400">F1 Score: {payload[2].value}%</div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Legend wrapperStyle={{ paddingTop: 10, fontSize: 12 }} />
            <Bar dataKey="precision" name="Precision" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            <Bar dataKey="recall" name="Recall" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            <Bar dataKey="f1_score" name="F1 Score" fill="#10b981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

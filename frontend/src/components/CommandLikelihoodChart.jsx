import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  CartesianGrid,
} from 'recharts';

export default function CommandLikelihoodChart({ scores, predictedCommand }) {
  if (!scores) return null;

  const data = Object.entries(scores)
    .map(([command, score]) => ({
      command,
      score: Number(score.toFixed(1)),
      isWinner: command === predictedCommand,
    }))
    .sort((a, b) => b.score - a.score);

  return (
    <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-base font-bold text-slate-900">Forward Log Likelihood Comparison</h3>
          <p className="text-xs text-slate-500">
            Evaluating ln P(X | λ_k) across all 10 candidate acoustic command models
          </p>
        </div>
        <span className="text-xs font-semibold px-2.5 py-1 rounded-md bg-slate-100 text-slate-700">
          Max: {data[0]?.command}
        </span>
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
            <XAxis type="number" tick={{ fontSize: 11 }} stroke="#94a3b8" />
            <YAxis
              type="category"
              dataKey="command"
              tick={{ fontSize: 12, fontWeight: 500 }}
              stroke="#64748b"
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const item = payload[0].payload;
                  return (
                    <div className="bg-slate-900 text-white p-2.5 rounded-lg text-xs shadow-lg">
                      <div className="font-bold capitalize">{item.command}</div>
                      <div className="text-slate-300">Log Likelihood: {item.score}</div>
                      {item.isWinner && (
                        <div className="text-emerald-400 font-semibold mt-1">★ Winner</div>
                      )}
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar dataKey="score" radius={[0, 4, 4, 0]}>
              {data.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.isWinner ? '#10b981' : '#6366f1'}
                  opacity={entry.isWinner ? 1.0 : 0.75}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

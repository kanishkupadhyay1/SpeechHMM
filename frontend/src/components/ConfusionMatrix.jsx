import React from 'react';

export default function ConfusionMatrix({ matrix, commands, accuracy, macroF1 }) {
  if (!matrix || !commands) return null;

  // Find max value in matrix for color intensity normalization
  let maxVal = 1;
  matrix.forEach((row) => {
    row.forEach((v) => {
      if (v > maxVal) maxVal = v;
    });
  });

  return (
    <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-sm overflow-x-auto">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 mb-6">
        <div>
          <h3 className="text-base font-bold text-slate-900">Held-Out Test Confusion Matrix</h3>
          <p className="text-xs text-slate-500">
            True Command (Rows) vs. HMM Predicted Command (Columns)
          </p>
        </div>
        <div className="flex gap-2">
          {accuracy !== undefined && (
            <span className="text-xs font-semibold px-2.5 py-1 rounded-md bg-emerald-50 text-emerald-700 border border-emerald-100">
              Acc: {(accuracy * 100).toFixed(1)}%
            </span>
          )}
          {macroF1 !== undefined && (
            <span className="text-xs font-semibold px-2.5 py-1 rounded-md bg-indigo-50 text-indigo-700 border border-indigo-100">
              Macro F1: {(macroF1 * 100).toFixed(1)}%
            </span>
          )}
        </div>
      </div>

      <div className="min-w-[500px]">
        {/* Table Header: Predicted Commands */}
        <div className="flex items-center text-xs font-bold text-slate-500 mb-2">
          <div className="w-16 shrink-0 text-right pr-2">True \ Pred</div>
          {commands.map((cmd) => (
            <div key={cmd} className="flex-1 text-center truncate px-1 text-[11px] uppercase">
              {cmd}
            </div>
          ))}
        </div>

        {/* Matrix Rows */}
        {matrix.map((row, rowIdx) => {
          const rowTotal = row.reduce((a, b) => a + b, 0);
          return (
            <div key={commands[rowIdx]} className="flex items-center my-1">
              <div className="w-16 shrink-0 text-right pr-2 font-bold text-xs text-slate-700 truncate capitalize">
                {commands[rowIdx]}
              </div>

              {row.map((val, colIdx) => {
                const isDiag = rowIdx === colIdx;
                const ratio = val / maxVal;
                const bgStyle = isDiag
                  ? `rgba(79, 70, 229, ${Math.max(0.12, ratio * 0.95)})`
                  : val > 0
                  ? `rgba(239, 68, 68, ${Math.min(0.7, ratio * 1.5)})`
                  : '#f8fafc';

                const textCol = isDiag && ratio > 0.4 ? 'text-white font-bold' : 'text-slate-800 font-medium';

                return (
                  <div
                    key={`${rowIdx}-${colIdx}`}
                    style={{ backgroundColor: bgStyle }}
                    title={`True: ${commands[rowIdx]}, Pred: ${commands[colIdx]} = ${val}`}
                    className={`flex-1 h-9 flex items-center justify-center text-xs rounded mx-0.5 transition-all hover:scale-105 hover:z-10 cursor-default ${textCol}`}
                  >
                    {val}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}

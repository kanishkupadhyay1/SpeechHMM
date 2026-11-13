import React, { useRef, useEffect } from 'react';

export default function MFCCChart({ mfccData }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!mfccData || !mfccData.mfcc_matrix || !canvasRef.current) return;

    const matrix = mfccData.mfcc_matrix; // shape: (T, D)
    const T = matrix.length;
    const D = matrix[0].length;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Compute min and max for normalization
    let min = Infinity;
    let max = -Infinity;
    for (let t = 0; t < T; t++) {
      for (let d = 0; d < D; d++) {
        const v = matrix[t][d];
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    const range = max - min || 1;

    const cellW = width / T;
    const cellH = height / D;

    // Color mapper (magma/inferno style)
    const getColor = (val) => {
      const norm = (val - min) / range;
      // Interpolate between dark indigo, magenta, and bright yellow
      const r = Math.min(255, Math.floor(norm * 255 * 1.2));
      const g = Math.min(255, Math.floor(Math.pow(norm, 2) * 230));
      const b = Math.min(255, Math.floor((1 - norm) * 150 + norm * 50));
      return `rgb(${r}, ${g}, ${b})`;
    };

    // Draw spectrogram cells (transposed: time on X, feature dimension on Y)
    for (let t = 0; t < T; t++) {
      for (let d = 0; d < D; d++) {
        ctx.fillStyle = getColor(matrix[t][d]);
        ctx.fillRect(t * cellW, (D - 1 - d) * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
  }, [mfccData]);

  if (!mfccData) return null;

  return (
    <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-base font-bold text-slate-900">Extracted MFCC Feature Matrix</h3>
          <p className="text-xs text-slate-500">
            {mfccData.time_frames} frames × {mfccData.feature_dim} dimensions (MFCC 1-13 + Δ + ΔΔ)
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-slate-500">
          <span className="w-2.5 h-2.5 rounded-full bg-indigo-900"></span> Low
          <span className="w-2.5 h-2.5 rounded-full bg-pink-500"></span> Mid
          <span className="w-2.5 h-2.5 rounded-full bg-amber-400"></span> High
        </div>
      </div>

      <div className="w-full rounded-xl overflow-hidden border border-slate-200 bg-slate-950 p-1">
        <canvas
          ref={canvasRef}
          width={800}
          height={200}
          className="w-full h-44 block rounded"
        />
      </div>

      <div className="flex justify-between items-center text-[11px] text-slate-400 mt-2 px-1">
        <span>0.0s (Onset)</span>
        <span>Time Evolution (10ms hop)</span>
        <span>1.0s (Offset)</span>
      </div>
    </div>
  );
}

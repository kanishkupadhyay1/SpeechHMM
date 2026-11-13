import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Upload, AlertCircle, Volume2, CheckCircle2 } from 'lucide-react';

export default function MicrophoneRecorder({ onAudioRecorded, isLoading }) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [permissionError, setPermissionError] = useState(null);
  const [audioLevel, setAudioLevel] = useState(0);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
    };
  }, []);

  const startRecording = async () => {
    setPermissionError(null);
    audioChunksRef.current = [];

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('MediaDevices API is not supported in this browser.');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      // Set up real-time audio level monitoring
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const analyser = audioCtx.createAnalyser();
      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);
      analyser.fftSize = 256;
      audioContextRef.current = audioCtx;
      analyserRef.current = analyser;

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const updateMeter = () => {
        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
          sum += dataArray[i];
        }
        const avg = sum / bufferLength;
        setAudioLevel(Math.min(100, Math.round((avg / 128) * 100)));
        animationFrameRef.current = requestAnimationFrame(updateMeter);
      };
      updateMeter();

      // Setup MediaRecorder
      const options = { mimeType: 'audio/webm' };
      let recorder;
      try {
        recorder = new MediaRecorder(stream, options);
      } catch (e) {
        recorder = new MediaRecorder(stream);
      }

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
        stream.getTracks().forEach((track) => track.stop());

        const mime = recorder.mimeType || 'audio/webm';
        const rawBlob = new Blob(audioChunksRef.current, { type: mime });

        // Convert audio to 16kHz mono WAV for backend HMM
        try {
          const wavBlob = await convertToWav(rawBlob);
          onAudioRecorded(wavBlob);
        } catch (convErr) {
          console.warn('WAV conversion fallback to raw blob:', convErr);
          onAudioRecorded(rawBlob);
        }
      };

      mediaRecorderRef.current = recorder;
      recorder.start(100);
      setIsRecording(true);
      setRecordingDuration(0);

      const startTimestamp = Date.now();
      timerRef.current = setInterval(() => {
        const elapsed = (Date.now() - startTimestamp) / 1000;
        setRecordingDuration(elapsed);
        // Automatically stop around 1.2 seconds for isolated command recognition
        if (elapsed >= 1.5) {
          stopRecording();
        }
      }, 50);
    } catch (err) {
      console.error('Microphone error:', err);
      setPermissionError(err.message || 'Microphone access denied or unavailable.');
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  };

  // Convert browser-recorded audio blob into standard 16,000 Hz Mono WAV
  const convertToWav = async (blob) => {
    const arrayBuffer = await blob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    // Resample / extract channel
    const targetSampleRate = 16000;
    const offlineCtx = new OfflineAudioContext(1, Math.ceil(audioBuffer.duration * targetSampleRate), targetSampleRate);
    const source = offlineCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineCtx.destination);
    source.start(0);

    const renderedBuffer = await offlineCtx.startRendering();
    const wavData = encodeWavPCM(renderedBuffer);
    return new Blob([wavData], { type: 'audio/wav' });
  };

  // PCM WAV encoder helper
  const encodeWavPCM = (buffer) => {
    const numChannels = 1;
    const sampleRate = buffer.sampleRate;
    const samples = buffer.getChannelData(0);
    const byteRate = sampleRate * numChannels * 2;
    const blockAlign = numChannels * 2;
    const dataSize = samples.length * 2;
    const bufferSize = 44 + dataSize;
    const arrayBuffer = new ArrayBuffer(bufferSize);
    const view = new DataView(arrayBuffer);

    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true); // 16-bit
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return arrayBuffer;
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      onAudioRecorded(file);
    }
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-sm">
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
        <div>
          <h3 className="text-lg font-bold text-slate-900">Acoustic Input Interface</h3>
          <p className="text-sm text-slate-500 mt-0.5">
            Record a single command (e.g., <span className="font-semibold text-indigo-600">yes, stop, go</span>) or upload a WAV file.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            accept=".wav,audio/wav"
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isRecording || isLoading}
            className="flex items-center space-x-2 px-4 py-2.5 rounded-xl border border-slate-200 bg-slate-50 hover:bg-slate-100 text-slate-700 text-sm font-medium transition-all"
          >
            <Upload className="w-4 h-4 text-slate-500" />
            <span>Upload WAV</span>
          </button>

          {!isRecording ? (
            <button
              onClick={startRecording}
              disabled={isLoading}
              className="flex items-center space-x-2 px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-semibold shadow-md shadow-indigo-100 transition-all disabled:opacity-50"
            >
              <Mic className="w-4 h-4" />
              <span>Record Command</span>
            </button>
          ) : (
            <button
              onClick={stopRecording}
              className="flex items-center space-x-2 px-6 py-2.5 rounded-xl bg-red-600 hover:bg-red-700 text-white text-sm font-semibold shadow-md shadow-red-100 transition-all animate-pulse"
            >
              <Square className="w-4 h-4 fill-white" />
              <span>Stop Recording ({recordingDuration.toFixed(1)}s)</span>
            </button>
          )}
        </div>
      </div>

      {isRecording && (
        <div className="mt-5 p-4 rounded-xl bg-red-50/50 border border-red-100 flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
            </span>
            <span className="text-xs font-semibold text-red-700 uppercase tracking-wider">Listening</span>
          </div>

          <div className="flex-1 bg-red-200/60 rounded-full h-2.5 overflow-hidden">
            <div
              className="bg-red-500 h-2.5 rounded-full transition-all duration-75"
              style={{ width: `${Math.max(5, audioLevel)}%` }}
            ></div>
          </div>
          <span className="text-xs font-mono text-red-600 font-medium">{recordingDuration.toFixed(1)}s / 1.5s</span>
        </div>
      )}

      {permissionError && (
        <div className="mt-4 p-3 rounded-xl bg-amber-50 border border-amber-200 text-amber-800 text-sm flex items-start gap-2">
          <AlertCircle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold">Microphone Note:</span> {permissionError}
            <div className="text-xs mt-1 text-amber-700">
              Ensure browser microphone permission is granted. On localhost, microphone access is enabled by standard browser policy.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

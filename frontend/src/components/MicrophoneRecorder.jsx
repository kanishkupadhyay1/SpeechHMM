import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Upload, AlertCircle, Volume2, CheckCircle2 } from 'lucide-react';

export default function MicrophoneRecorder({ onAudioRecorded, isLoading }) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [permissionError, setPermissionError] = useState(null);
  const [audioLevel, setAudioLevel] = useState(0);

  const mediaStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const scriptProcessorRef = useRef(null);
  const mediaSourceRef = useRef(null);
  const recordedSamplesRef = useRef([]);
  const timerRef = useRef(null);
  const animationFrameRef = useRef(null);
  const fileInputRef = useRef(null);

  // Fallback MediaRecorder
  const mediaRecorderRef = useRef(null);
  const rawChunksRef = useRef([]);

  useEffect(() => {
    return () => {
      cleanupAudio();
    };
  }, []);

  const cleanupAudio = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (scriptProcessorRef.current) {
      try {
        scriptProcessorRef.current.disconnect();
      } catch (e) {}
      scriptProcessorRef.current = null;
    }
    if (mediaSourceRef.current) {
      try {
        mediaSourceRef.current.disconnect();
      } catch (e) {}
      mediaSourceRef.current = null;
    }
    if (mediaStreamRef.current) {
      try {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      } catch (e) {}
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      try {
        audioContextRef.current.close();
      } catch (e) {}
      audioContextRef.current = null;
    }
  };

  const startRecording = async () => {
    setPermissionError(null);
    recordedSamplesRef.current = [];
    rawChunksRef.current = [];

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
          'Microphone API is not supported or not accessible. Note: Browsers require HTTPS or localhost for microphone access.'
        );
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      mediaStreamRef.current = stream;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioCtx();
      audioContextRef.current = audioCtx;

      // Resume context if suspended (common in modern browser autoplay policies)
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }

      const source = audioCtx.createMediaStreamSource(stream);
      mediaSourceRef.current = source;

      // Analyser for real-time VU meter
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

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

      // Direct PCM sampling via ScriptProcessorNode (4096 buffer size)
      // Captures uncompressed float32 samples directly from microphone stream
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      scriptProcessorRef.current = processor;
      processor.onaudioprocess = (e) => {
        const channelData = e.inputBuffer.getChannelData(0);
        recordedSamplesRef.current.push(new Float32Array(channelData));
      };

      source.connect(processor);
      // Connect to destination through a silent gain node to keep processor active
      const silentGain = audioCtx.createGain();
      silentGain.gain.value = 0;
      processor.connect(silentGain);
      silentGain.connect(audioCtx.destination);

      // Concurrent MediaRecorder as secondary fallback
      try {
        const recorder = new MediaRecorder(stream);
        recorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) rawChunksRef.current.push(e.data);
        };
        mediaRecorderRef.current = recorder;
        recorder.start(100);
      } catch (recErr) {
        // MediaRecorder optional; direct PCM will handle it
      }

      setIsRecording(true);
      setRecordingDuration(0);

      const startTimestamp = Date.now();
      timerRef.current = setInterval(() => {
        const elapsed = (Date.now() - startTimestamp) / 1000;
        setRecordingDuration(elapsed);
        if (elapsed >= 1.5) {
          stopRecording();
        }
      }, 50);
    } catch (err) {
      console.error('Microphone error:', err);
      let msg = err.message || 'Microphone access denied or unavailable.';
      if (err.name === 'NotAllowedError') {
        msg = 'Microphone permission was denied. Please allow microphone access in your browser settings.';
      } else if (err.name === 'SecurityError' || (window.location.protocol === 'http:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1')) {
        msg = 'Microphone access requires HTTPS or localhost by browser security policies. Please use HTTPS or upload a WAV file.';
      }
      setPermissionError(msg);
      setIsRecording(false);
      cleanupAudio();
    }
  };

  const stopRecording = () => {
    if (!isRecording) return;
    setIsRecording(false);

    const sourceSampleRate = audioContextRef.current ? audioContextRef.current.sampleRate : 16000;
    const pcmChunks = [...recordedSamplesRef.current];

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      try {
        mediaRecorderRef.current.stop();
      } catch (e) {}
    }

    cleanupAudio();

    // Process captured PCM audio into standard 16,000 Hz 16-bit Mono WAV
    setTimeout(async () => {
      try {
        let totalSamplesCount = 0;
        for (const chunk of pcmChunks) {
          totalSamplesCount += chunk.length;
        }

        if (totalSamplesCount > 1000) {
          // Merge all float32 chunks
          const merged = new Float32Array(totalSamplesCount);
          let offset = 0;
          for (const chunk of pcmChunks) {
            merged.set(chunk, offset);
            offset += chunk.length;
          }

          // Resample to 16,000 Hz if needed
          const resampled = resampleTo16k(merged, sourceSampleRate);
          // Encode to standard 16-bit PCM WAV
          const wavBlob = encodeWavFromFloat32(resampled, 16000);
          onAudioRecorded(wavBlob);
          return;
        }

        // Secondary fallback: if PCM was empty, attempt MediaRecorder chunks
        if (rawChunksRef.current.length > 0) {
          const rawBlob = new Blob(rawChunksRef.current, { type: 'audio/webm' });
          const wavBlob = await convertBlobToWavFallback(rawBlob);
          onAudioRecorded(wavBlob);
          return;
        }

        throw new Error('No audio was captured. Please check your microphone and speak clearly.');
      } catch (err) {
        console.error('Audio processing error:', err);
        setPermissionError(`Audio recording error: ${err.message}. Try uploading a WAV file instead.`);
      }
    }, 150);
  };

  // Resample float32 array to 16,000 Hz
  const resampleTo16k = (audioData, origSampleRate) => {
    if (origSampleRate === 16000) return audioData;
    const targetSampleRate = 16000;
    const ratio = origSampleRate / targetSampleRate;
    const newLength = Math.round(audioData.length / ratio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
      let accum = 0;
      let count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < audioData.length; i++) {
        accum += audioData[i];
        count++;
      }
      result[offsetResult] = count > 0 ? accum / count : 0;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return result;
  };

  // Direct 16-bit PCM WAV encoder
  const encodeWavFromFloat32 = (samples, sampleRate = 16000) => {
    const numChannels = 1;
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
    return new Blob([arrayBuffer], { type: 'audio/wav' });
  };

  // Fallback converter for MediaRecorder blob
  const convertBlobToWavFallback = async (blob) => {
    const arrayBuffer = await blob.arrayBuffer();
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioCtx();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const channelData = audioBuffer.getChannelData(0);
    const resampled = resampleTo16k(channelData, audioBuffer.sampleRate);
    return encodeWavFromFloat32(resampled, 16000);
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

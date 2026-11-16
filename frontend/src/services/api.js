import axios from 'axios';

// Dynamically determine backend URL:
// 1. Explicit environment variable VITE_API_URL if configured during deployment
// 2. Relative path '' when deployed to any production host / container
// 3. Fallback to http://localhost:8000 when developing locally
export const getApiBase = () => {
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    // When served directly from FastAPI (single-service deployment), port is 8000
    if (window.location.port === '8000') {
      return '';
    }
    // When running in Vite dev server on localhost, point to FastAPI port 8000
    if (window.location.port === '5173' && (hostname === 'localhost' || hostname === '127.0.0.1')) {
      return 'http://localhost:8000';
    }
    // Deployed to any cloud/remote URL (Vercel, Render, Railway, etc.): use relative path
    return '';
  }
  return 'http://localhost:8000';
};

const API_BASE = getApiBase();

const client = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export const getHealth = async () => {
  const res = await client.get('/');
  return res.data;
};

export const getCommands = async () => {
  const res = await client.get('/api/commands');
  return res.data;
};

export const getMetrics = async () => {
  const res = await client.get('/api/metrics');
  return res.data;
};

export const getConfusionMatrix = async () => {
  const res = await client.get('/api/confusion-matrix');
  return res.data;
};

export const getModelAnalysis = async () => {
  const res = await client.get('/api/model-analysis');
  return res.data;
};

export const recognizeAudio = async (audioBlob) => {
  const formData = new FormData();
  formData.append('file', audioBlob, 'command.wav');
  const res = await client.post('/api/recognize', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export const getMFCCData = async (audioBlob) => {
  const formData = new FormData();
  formData.append('file', audioBlob, 'command.wav');
  const res = await client.post('/api/mfcc', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export default client;

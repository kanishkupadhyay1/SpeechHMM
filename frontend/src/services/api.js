import axios from 'axios';

const API_BASE = 'http://localhost:8000';

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

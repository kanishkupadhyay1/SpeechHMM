import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import LiveRecognition from './pages/LiveRecognition';
import ModelAnalysis from './pages/ModelAnalysis';
import About from './pages/About';

export default function App() {
  const [activePage, setActivePage] = useState('dashboard');

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans">
      <Navbar activePage={activePage} setActivePage={setActivePage} />

      <main className="flex-1 pb-16">
        {activePage === 'dashboard' && <Dashboard setActivePage={setActivePage} />}
        {activePage === 'live' && <LiveRecognition />}
        {activePage === 'analysis' && <ModelAnalysis />}
        {activePage === 'about' && <About />}
      </main>

      <footer className="border-t border-slate-200 bg-white py-6">
        <div className="max-w-7xl mx-auto px-4 text-center text-xs text-slate-500">
          AI-Powered Speech Command Recognition Using Continuous Gaussian Hidden Markov Models | Google Speech Commands v0.02
        </div>
      </footer>
    </div>
  );
}

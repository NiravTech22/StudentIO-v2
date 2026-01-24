import { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Activity, Zap, Server, Cpu } from 'lucide-react';
import { ChatInterface } from '../frontend/ChatInterface';
import { API_BASE_URL } from './api_client';

// ============================================================================
// Types
// ============================================================================

interface CoreTelemetry {
  timestamp: string;
  activations: number[]; // size 128 (hidden state)
}

// ============================================================================
// Components
// ============================================================================

function NeuralCoreVisualizer({ data }: { data: number[] }) {
  // Render a 8x16 grid of activations
  return (
    <div className="relative group perspective-1000">
      <div className="absolute inset-0 bg-indigo-500/5 blur-3xl rounded-full transition-opacity duration-1000 animate-pulse" />

      <div className="relative bg-slate-900/80 backdrop-blur-xl border border-white/10 rounded-[2rem] p-8 shadow-2xl ring-1 ring-white/5 transform transition-all duration-500 hover:rotate-x-2">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-500/20 rounded-lg">
              <Cpu size={20} className="text-indigo-400" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-slate-200 uppercase tracking-widest">LSTM Core State</h2>
              <div className="text-[10px] text-slate-500 font-mono">128-DIMENSIONAL HIDDEN VECTOR</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_10px_#10b981]" />
            <span className="text-xs font-mono text-emerald-400">ACTIVE</span>
          </div>
        </div>

        {/* The Grid */}
        <div className="grid grid-cols-16 gap-1.5 w-full mx-auto">
          {Array.from({ length: 128 }).map((_, i) => {
            const val = data[i] || 0;
            // Normalize vaguely for visualization (activations roughly -1 to 1)
            const intensity = (Math.tanh(val) + 1) / 2;
            return (
              <motion.div
                key={i}
                initial={false}
                animate={{
                  backgroundColor: `rgba(99, 102, 241, ${intensity})`,
                  scale: 0.8 + (intensity * 0.4),
                  boxShadow: `0 0 ${intensity * 10}px rgba(99, 102, 241, ${intensity * 0.8})`
                }}
                transition={{ duration: 0.2 }}
                className="w-full aspect-square rounded-[2px] opacity-80"
                title={`Neuron ${i}: ${val.toFixed(3)}`}
              />
            );
          })}
        </div>

        {/* Footer Metrics */}
        <div className="mt-8 pt-6 border-t border-white/5 flex justify-between text-[10px] font-mono text-slate-500 uppercase">
          <div className="flex gap-4">
            <span>Mean Act: <span className="text-indigo-300">{(data.reduce((a, b) => a + Math.abs(b), 0) / (data.length || 1)).toFixed(3)}</span></span>
            <span>Active Units: <span className="text-indigo-300">{data.filter(x => Math.abs(x) > 0.1).length}</span></span>
          </div>
          <div>
            Flux.jl / CPU
          </div>
        </div>
      </div>
    </div>
  );
}

export default function StudentIODashboard() {
  const [telemetry, setTelemetry] = useState<CoreTelemetry>({ timestamp: '', activations: [] });
  const [error, setError] = useState<string | null>(null);

  // Dummy student ID for the chat interface - the backend now ignores it but the component requires it
  const DUMMY_STUDENT = 1;

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/telemetry`);
        if (!res.ok) throw new Error('Network response was not ok');
        const data = await res.json();
        setTelemetry(data);
        setError(null);
      } catch (e) {
        // Keep silent on polling errors to avoid flickering, just don't update
        // setError("Connecting...");
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 200); // 5Hz update rate for smooth visuals
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#05050A] text-slate-200 p-6 md:p-12 font-sans selection:bg-indigo-500/30 overflow-hidden flex flex-col">
      {/* Header */}
      <header className="flex justify-between items-center mb-12 z-10 shrink-0">
        <div className="flex items-center gap-5">
          <div className="relative group cursor-pointer">
            <div className="absolute inset-0 bg-indigo-600 rounded-2xl blur opacity-40 group-hover:opacity-60 transition-opacity" />
            <div className="relative p-3.5 bg-indigo-600 rounded-2xl shadow-xl border border-white/10">
              <Brain size={32} className="text-white" />
            </div>
          </div>
          <div>
            <h1 className="text-3xl font-black tracking-tighter italic leading-none text-white flex gap-1">
              STUDENT<span className="text-indigo-500">IO</span> <span className="text-indigo-900 not-italic font-normal">v2</span>
            </h1>
            <div className="flex items-center gap-2 mt-1.5">
              <span className="text-[10px] font-bold tracking-[0.2em] text-indigo-400/80 uppercase">Flux Neural Interface</span>
              <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-white/5 text-slate-400 border border-white/5">CPU MODE</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs font-bold uppercase tracking-wider text-slate-500">
          <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/5">
            <Server size={14} className={error ? "text-red-500" : "text-emerald-500"} />
            <span className={error ? "text-red-400" : "text-emerald-400"}>{error ? "OFFLINE" : "SYSTEM ONLINE"}</span>
          </div>
        </div>
      </header>

      {/* Main Content: Split View */}
      <div className="flex-1 flex flex-col lg:flex-row gap-8 lg:gap-12 min-h-0">

        {/* Left: Neural Core Visualization */}
        <div className="flex-1 flex flex-col justify-center max-w-4xl mx-auto w-full">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="w-full"
          >
            <NeuralCoreVisualizer data={telemetry.activations} />
          </motion.div>
        </div>

        {/* Right: Chat Interface */}
        <div className="lg:w-[450px] shrink-0 h-[600px] lg:h-auto flex flex-col">
          <ChatInterface studentId={DUMMY_STUDENT} />
        </div>

      </div>
    </div>
  );
}

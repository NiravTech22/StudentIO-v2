import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Brain, Activity, Zap } from 'lucide-react';

export default function StudentIODashboard() {
  const [telemetry, setTelemetry] = useState([]);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch('http://localhost:8080/api/telemetry');
        const data = await res.json();
        setTelemetry(data.students);
      } catch (e) { console.error("Server Offline"); }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#060609] text-slate-200 p-10 font-sans">
      <header className="flex justify-between items-center mb-16">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-indigo-600 rounded-2xl shadow-lg shadow-indigo-500/40">
            <Brain size={28} className="text-white" />
          </div>
          <h1 className="text-2xl font-black tracking-tighter italic">STUDENT<span className="text-indigo-500">IO</span></h1>
        </div>
        <div className="px-4 py-2 bg-emerald-500/10 border border-emerald-500/20 rounded-full text-emerald-400 text-xs font-bold uppercase tracking-widest flex items-center gap-2">
          <Zap size={14} /> CUDA Engine: Synchronized
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {telemetry.map((s) => (
          <motion.div key={s.id} layout className="bg-white/5 border border-white/10 rounded-[2.5rem] p-8 backdrop-blur-xl">
            <div className="flex justify-between items-start mb-6">
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-tighter">Student Instance {s.id}</span>
              <span className="text-[10px] font-mono text-indigo-400">t+{s.timestep}</span>
            </div>

            <div className="grid grid-cols-10 gap-1.5">
              {s.belief.map((val, i) => (
                <div
                  key={i}
                  className="w-full aspect-square rounded-[2px] transition-all duration-700"
                  style={{
                    backgroundColor: `rgba(99, 102, 241, ${val})`,
                    boxShadow: `0 0 ${s.variance[i] * 15}px rgba(99, 102, 241, 0.4)`
                  }}
                />
              ))}
            </div>
            <div className="mt-6 pt-6 border-t border-white/5 flex justify-between items-center">
              <Activity size={16} className="text-slate-600" />
              <span className="text-[10px] text-slate-400 font-bold uppercase">RNN State Inference</span>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
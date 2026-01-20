import { useEffect, useState, forwardRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Activity, Zap, AlertTriangle, Loader2 } from 'lucide-react';

// ============================================================================
// Types
// ============================================================================

/**
 * Matches the StudentState_C structure from Julia backend (student_io_core.jl)
 * and the specific API response shape from /api/telemetry.
 */
interface StudentState {
  id: number;
  belief: number[];       // size 50, float32 (belief_mean)
  variance: number[];     // size 50, float32 (belief_var)
  timestep: number;
}

interface TelemetryResponse {
  timestamp: number;
  students: StudentState[];
}

// ============================================================================
// Components
// ============================================================================

const StudentCard = forwardRef<HTMLDivElement, { student: StudentState }>(({ student }, ref) => (
  <motion.div
    ref={ref}
    layout
    className="bg-white/10 border border-white/20 rounded-[2.5rem] p-8 backdrop-blur-xl hover:bg-white/15 hover:border-indigo-500/50 transition-all duration-300 group"
  >
    <div className="flex justify-between items-start mb-6">
      <span className="text-xs font-bold text-slate-300 uppercase tracking-wider group-hover:text-white transition-colors">Student Instance {student.id}</span>
      <span className="text-xs font-mono text-indigo-400 group-hover:text-indigo-300 transition-colors">t+{student.timestep}</span>
    </div>

    <div className="grid grid-cols-10 gap-2 mb-6">
      {student.belief.map((val, i) => (
        <div
          key={`${student.id}-belief-${i}`}
          className="w-full aspect-square rounded-[2px] transition-all duration-700"
          style={{
            backgroundColor: `rgba(99, 102, 241, ${val})`, // Indigo base
            opacity: Math.max(0.2, val), // Ensure at least some visibility
            boxShadow: `0 0 ${(student.variance[i] || 0) * 10}px rgba(99, 102, 241, 0.5)`
          }}
          title={`Dim ${i}: ${val.toFixed(2)} (σ²: ${(student.variance[i] || 0).toFixed(2)})`}
        />
      ))}
    </div>

    <div className="pt-6 border-t border-white/10 flex justify-between items-center">
      <div className="flex items-center gap-2">
        <Activity size={16} className="text-indigo-400" />
        <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">RNN State Inference</span>
      </div>
      <div className="text-[10px] text-slate-400 font-mono">
        Active Dimensions: <span className="text-indigo-300">{student.belief.filter(b => b > 0.1).length}</span>
      </div>
    </div>
  </motion.div>
));

StudentCard.displayName = "StudentCard";

export default function StudentIODashboard() {
  const [telemetry, setTelemetry] = useState<StudentState[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    let isMounted = true;
    const POLLING_RATE_MS = 1000;

    const fetchData = async () => {
      try {
        const res = await fetch('http://localhost:8080/api/telemetry');

        if (!res.ok) {
          throw new Error(`Server responded with ${res.status}`);
        }

        const data: TelemetryResponse = await res.json();

        if (isMounted) {
          // Simple validation to ensure we have the array we expect
          if (Array.isArray(data.students)) {
            setTelemetry(data.students);
            setError(null);
          } else {
            console.warn("Received malformed data:", data);
            // Don't throw here, just don't update if it's partial garbage but keep old state
          }
          setIsLoading(false);
        }
      } catch (e) {
        if (isMounted) {
          console.error("Connection Error:", e);
          setError(e instanceof Error ? e.message : "Checking connection...");
          // Keep showing potentially stale data if we lose connection briefly? 
          // Or show error state? Let's show error overlay but keep data if user wants.
        }
      }
    };

    // Initial fetch
    fetchData();

    const interval = setInterval(fetchData, POLLING_RATE_MS);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-10 font-sans selection:bg-indigo-500/30">
      <header className="flex justify-between items-center mb-16 relative z-10">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-indigo-600 rounded-2xl shadow-lg shadow-indigo-500/40">
            <Brain size={28} className="text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-black tracking-tighter italic leading-none text-white">
              STUDENT<span className="text-indigo-400">IO</span>
            </h1>
            <div className="text-[10px] font-bold tracking-widest text-slate-400 uppercase mt-1">
              Neural Telemetry Dashboard
            </div>
          </div>
        </div>

        <div className={`
          px-4 py-2 rounded-full text-xs font-bold uppercase tracking-widest flex items-center gap-2 border transition-colors backdrop-blur-md
          ${error
            ? 'bg-red-500/10 border-red-500/20 text-red-400'
            : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
          }
        `}>
          {error ? (
            <>
              <AlertTriangle size={14} />
              <span>Backend Offline</span>
            </>
          ) : (
            <>
              <Zap size={14} />
              <span>CUDA Synchronized</span>
            </>
          )}
        </div>
      </header>

      {/* Main Content Area */}
      {isLoading && !telemetry.length && !error ? (
        <div className="flex h-[50vh] w-full justify-center items-center flex-col gap-4 text-slate-500">
          <Loader2 className="animate-spin text-indigo-500" size={32} />
          <span className="text-xs uppercase tracking-widest font-bold">Establishing Uplink...</span>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 pb-20">
          <AnimatePresence mode="popLayout">
            {telemetry.map((s) => (
              <StudentCard key={s.id} student={s} />
            ))}
          </AnimatePresence>

          {/* Empty State / Error Fallback Display */}
          {telemetry.length === 0 && !isLoading && (
            <div className="col-span-full py-20 text-center border border-dashed border-slate-800 rounded-3xl bg-white/5">
              <div className="text-slate-400 font-mono text-sm">No Active Student Instances Found</div>
              {error && <div className="text-red-400 text-xs mt-2 font-mono">{error}</div>}
            </div>
          )}
        </div>
      )}

      {/* Connectivity Status Footer (Visible on Error) */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-8 right-8 max-w-sm bg-red-950/90 border border-red-500/30 p-4 rounded-xl backdrop-blur text-red-100 text-xs font-mono shadow-2xl"
        >
          <div className="font-bold mb-1 flex items-center gap-2">
            <Loader2 size={12} className="animate-spin" />
            Reconnecting to Neural Backend...
          </div>
          <div className="opacity-70 truncate">
            Target: http://localhost:8080/api/telemetry
          </div>
        </motion.div>
      )}
    </div>
  );
}

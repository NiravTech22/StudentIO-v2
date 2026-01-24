import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Brain, Activity, TrendingUp, Zap, Target, AlertTriangle, } from 'lucide-react';

interface Props {
    sessionId: string;
    apiUrl: string;
}

interface StateData {
    step: number;
    mastery: number[];
    misconceptions: number[];
    abstractions: number[];
    uncertainty: number;
}

export default function POMDPVisualizer({ sessionId, apiUrl }: Props) {
    const [trajectory, setTrajectory] = useState<StateData[]>([]);
    const [currentStep, setCurrentStep] = useState<any>(null);
    const [diagnostics, setDiagnostics] = useState<any>(null);
    const [config, setConfig] = useState<any>(null);

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const response = await fetch(`${apiUrl}/api/session/${sessionId}/history`);
                const data = await response.json();

                if (data.trajectory) {
                    setTrajectory(data.trajectory);
                }
            } catch (error) {
                console.error('Failed to fetch history:', error);
            }
        };

        const fetchDiagnostics = async () => {
            try {
                const response = await fetch(`${apiUrl}/api/session/${sessionId}/diagnostics`);
                const data = await response.json();
                setDiagnostics(data);
            } catch (error) {
                console.error('Failed to fetch diagnostics:', error);
            }
        };

        fetchHistory();
        fetchDiagnostics();

        const interval = setInterval(() => {
            fetchHistory();
            fetchDiagnostics();
        }, 2000);

        return () => clearInterval(interval);
    }, [sessionId, apiUrl]);

    if (!trajectory.length) {
        return (
            <div className="flex items-center justify-center h-96 text-text-secondary">
                <div className="text-center">
                    <Brain className="w-16 h-16 mx-auto mb-4 opacity-20" />
                    <p>No trajectory data yet. Start interacting with the system.</p>
                </div>
            </div>
        );
    }

    const latestState = trajectory[trajectory.length - 1];
    const meanMastery = latestState.mastery.reduce((a, b) => a + b, 0) / latestState.mastery.length;
    const hasMisconceptions = latestState.misconceptions.some(m => m > 0.3);

    return (
        <div className="space-y-6">
            {/* Header Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <StatCard
                    icon={Brain}
                    label="Steps"
                    value={trajectory.length.toString()}
                    color="primary"
                />
                <StatCard
                    icon={Target}
                    label="Mean Mastery"
                    value={`${(meanMastery * 100).toFixed(1)}%`}
                    color="success"
                />
                <StatCard
                    icon={AlertTriangle}
                    label="Misconceptions"
                    value={hasMisconceptions ? "Active" : "None"}
                    color={hasMisconceptions ? "warning" : "success"}
                />
                <StatCard
                    icon={Activity}
                    label="Uncertainty"
                    value={latestState.uncertainty.toFixed(3)}
                    color="secondary"
                />
            </div>

            {/* Latent State Decomposition */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass-strong rounded-3xl p-6 border border-white/10"
            >
                <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
                    <Brain className="text-primary" size={24} />
                    Latent State: x<sub>t</sub> ∈ ℝ<sup>64</sup>
                </h3>

                {/* Mastery (40 dims) */}
                <div className="mb-6">
                    <div className="flex items-center justify-between mb-3">
                        <h4 className="text-sm font-semibold text-text-primary flex items-center gap-2">
                            <TrendingUp size={16} className="text-success" />
                            Mastery (40 dimensions)
                        </h4>
                        <span className="text-xs text-text-tertiary">Mean: {(meanMastery * 100).toFixed(1)}%</span>
                    </div>
                    <div className="grid grid-cols-20 gap-1">
                        {latestState.mastery.map((value, i) => (
                            <div
                                key={i}
                                className="aspect-square rounded-sm transition-all"
                                style={{
                                    backgroundColor: `rgba(16, 185, 129, ${value})`,
                                    boxShadow: value > 0.7 ? `0 0 8px rgba(16, 185, 129, ${value})` : 'none'
                                }}
                                title={`Mastery[${i}]: ${value.toFixed(3)}`}
                            />
                        ))}
                    </div>
                </div>

                {/* Misconceptions (16 dims) */}
                <div className="mb-6">
                    <div className="flex items-center justify-between mb-3">
                        <h4 className="text-sm font-semibold text-text-primary flex items-center gap-2">
                            <AlertTriangle size={16} className="text-warning" />
                            Misconceptions (16 dimensions)
                        </h4>
                    </div>
                    <div className="grid grid-cols-16 gap-1.5">
                        {latestState.misconceptions.map((value, i) => (
                            <div
                                key={i}
                                className="aspect-square rounded-sm transition-all"
                                style={{
                                    backgroundColor: `rgba(245, 158, 11, ${value})`,
                                    boxShadow: value > 0.3 ? `0 0 8px rgba(245, 158, 11, ${value})` : 'none'
                                }}
                                title={`Misconception[${i}]: ${value.toFixed(3)}`}
                            />
                        ))}
                    </div>
                </div>

                {/* Abstractions (8 dims) */}
                <div>
                    <div className="flex items-center justify-between mb-3">
                        <h4 className="text-sm font-semibold text-text-primary flex items-center gap-2">
                            <Zap size={16} className="text-secondary" />
                            Abstractions (8 dimensions)
                        </h4>
                    </div>
                    <div className="grid grid-cols-8 gap-2">
                        {latestState.abstractions.map((value, i) => {
                            const normalized = Math.tanh(value); // Map to [-1, 1]
                            const intensity = Math.abs(normalized);
                            const color = normalized > 0 ? '139, 92, 246' : '239, 68, 68'; // purple or red

                            return (
                                <div
                                    key={i}
                                    className="aspect-square rounded-lg flex items-center justify-center text-xs font-mono transition-all"
                                    style={{
                                        backgroundColor: `rgba(${color}, ${intensity * 0.3})`,
                                        border: `2px solid rgba(${color}, ${intensity})`,
                                        color: intensity > 0.5 ? 'white' : '#94A3B8'
                                    }}
                                    title={`Abstraction[${i}]: ${value.toFixed(3)}`}
                                >
                                    {value >= 0 ? '+' : ''}{value.toFixed(1)}
                                </div>
                            );
                        })}
                    </div>
                </div>
            </motion.div>

            {/* Uncertainty Trajectory */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="glass-strong rounded-3xl p-6 border border-white/10"
            >
                <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
                    <Activity className="text-primary" size={24} />
                    Belief Uncertainty Trajectory
                </h3>

                <div className="relative h-32">
                    <svg className="w-full h-full" viewBox="0 0 600 100" preserveAspectRatio="none">
                        {/* Grid lines */}
                        {[0, 0.25, 0.5, 0.75, 1].map((y, i) => (
                            <line
                                key={i}
                                x1="0"
                                y1={y * 100}
                                x2="600"
                                y2={y * 100}
                                stroke="rgba(255,255,255,0.05)"
                                strokeWidth="1"
                            />
                        ))}

                        {/* Uncertainty line */}
                        <polyline
                            points={trajectory.map((state, i) => {
                                const x = (i / (trajectory.length - 1 || 1)) * 600;
                                const y = (1 - Math.min(state.uncertainty, 1)) * 100;
                                return `${x},${y}`;
                            }).join(' ')}
                            fill="none"
                            stroke="rgb(99, 102, 241)"
                            strokeWidth="2"
                        />

                        {/* Fill under curve */}
                        <polygon
                            points={
                                trajectory.map((state, i) => {
                                    const x = (i / (trajectory.length - 1 || 1)) * 600;
                                    const y = (1 - Math.min(state.uncertainty, 1)) * 100;
                                    return `${x},${y}`;
                                }).join(' ') + ' 600,100 0,100'
                            }
                            fill="rgba(99, 102, 241, 0.1)"
                        />
                    </svg>

                    {/* Y-axis labels */}
                    <div className="absolute left-0 top-0 bottom-0 flex flex-col justify-between text-xs text-text-tertiary font-mono -ml-12">
                        <span>1.0</span>
                        <span>0.5</span>
                        <span>0.0</span>
                    </div>
                </div>

                {diagnostics && diagnostics.uncertainty_collapsed && (
                    <div className="mt-4 p-3 bg-warning/10 border border-warning/30 rounded-lg flex items-center gap-2 text-sm">
                        <AlertTriangle size={16} className="text-warning" />
                        <span className="text-warning font-semibold">Warning:</span>
                        <span className="text-text-secondary">Uncertainty collapse detected - model may be overconfident</span>
                    </div>
                )}
            </motion.div>

            {/* Belief Drift */}
            {diagnostics && diagnostics.belief_drift && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="glass-strong rounded-3xl p-6 border border-white/10"
                >
                    <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                        <TrendingUp className="text-secondary" size={24} />
                        Belief Drift ||h<sub>t</sub> - h<sub>t-1</sub>||
                    </h3>

                    <div className="space-y-2">
                        {diagnostics.belief_drift.slice(-20).map((drift: number, i: number) => {
                            const maxDrift = Math.max(...diagnostics.belief_drift);
                            const percentage = (drift / maxDrift) * 100;

                            return (
                                <div key={i} className="flex items-center gap-3">
                                    <span className="text-xs font-mono text-text-tertiary w-12">t={trajectory.length - 20 + i}</span>
                                    <div className="flex-1 h-6 bg-bg-tertiary rounded-full overflow-hidden">
                                        <motion.div
                                            initial={{ width: 0 }}
                                            animate={{ width: `${percentage}%` }}
                                            className="h-full bg-gradient-to-r from-secondary to-accent rounded-full"
                                        />
                                    </div>
                                    <span className="text-xs font-mono text-text-secondary w-16">{drift.toFixed(3)}</span>
                                </div>
                            );
                        })}
                    </div>

                    <div className="mt-4 text-xs text-text-tertiary">
                        Mean drift: {diagnostics.mean_drift?.toFixed(4)} | Spikes indicate significant belief updates from surprising observations
                    </div>
                </motion.div>
            )}
        </div>
    );
}

function StatCard({ icon: Icon, label, value, color }: {
    icon: any;
    label: string;
    value: string;
    color: string;
}) {
    const colorClasses = {
        primary: 'text-primary bg-primary/20 border-primary/30',
        success: 'text-success bg-success/20 border-success/30',
        warning: 'text-warning bg-warning/20 border-warning/30',
        secondary: 'text-secondary bg-secondary/20 border-secondary/30'
    };

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="glass rounded-2xl p-4 border border-white/10 hover:border-primary/30 transition-all"
        >
            <div className={`inline-flex p-2 rounded-lg mb-3 ${colorClasses[color as keyof typeof colorClasses]}`}>
                <Icon size={20} />
            </div>
            <div className="text-2xl font-bold text-text-primary mb-1">{value}</div>
            <div className="text-xs text-text-tertiary uppercase tracking-wider">{label}</div>
        </motion.div>
    );
}

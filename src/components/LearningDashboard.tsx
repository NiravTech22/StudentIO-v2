import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, Brain, Clock, Target, Award, Zap } from 'lucide-react';

interface Props {
    studentId: string;
    apiUrl: string;
}

export default function LearningDashboard({ studentId, apiUrl }: Props) {
    const [profile, setProfile] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchProfile = async () => {
            try {
                const response = await fetch(`${apiUrl}/api/profile/${studentId}`);
                const data = await response.json();
                setProfile(data);
            } catch (error) {
                console.error('Failed to fetch profile:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchProfile();
    }, [studentId, apiUrl]);

    if (loading) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="spinner" />
            </div>
        );
    }

    const stats = profile?.stats || {};
    const learningProfile = profile?.profile || {};

    const topicsData = Object.entries(stats.topic_distribution || {}).slice(0, 5);

    return (
        <div className="max-w-6xl mx-auto space-y-6">
            {/* Header */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass-strong rounded-3xl p-8 border border-white/10"
            >
                <div className="flex items-center justify-between">
                    <div>
                        <h2 className="text-3xl font-black mb-2">
                            Your Learning <span className="gradient-text">Journey</span>
                        </h2>
                        <p className="text-text-secondary">Track your progress and insights</p>
                    </div>

                    <div className="p-4 bg-gradient-primary rounded-2xl shadow-glow-primary-sm">
                        <Brain size={40} className="text-white" />
                    </div>
                </div>
            </motion.div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                    { label: 'Questions Asked', value: stats.total_questions || 0, icon: Brain, color: 'primary' },
                    { label: 'Avg Confidence', value: `${Math.round((stats.avg_confidence || 0) * 100)}%`, icon: Target, color: 'success' },
                    { label: 'Helpful Rate', value: `${Math.round(stats.helpful_percentage || 0)}%`, icon: Award, color: 'accent' },
                    { label: 'Learning Style', value: learningProfile.style || 'Balanced', icon: Zap, color: 'secondary' }
                ].map((stat, index) => {
                    const Icon = stat.icon;

                    return (
                        <motion.div
                            key={stat.label}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.1 }}
                            className="glass rounded-2xl p-6 border border-white/10 hover:border-primary/30 transition-all group"
                        >
                            <div className="flex items-start justify-between mb-4">
                                <div className={`p-2 rounded-lg bg-${stat.color}/20`}>
                                    <Icon size={20} className={`text-${stat.color}`} />
                                </div>
                            </div>

                            <div className="text-3xl font-bold mb-1 text-text-primary">
                                {stat.value}
                            </div>

                            <div className="text-sm text-text-tertiary">
                                {stat.label}
                            </div>
                        </motion.div>
                    );
                })}
            </div>

            {/* Topic Distribution */}
            {topicsData.length > 0 && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                    className="glass-strong rounded-3xl p-8 border border-white/10"
                >
                    <h3 className="text-xl font-bold mb-6 flex items-center gap-2 text-text-primary">
                        <TrendingUp size={24} className="text-primary" />
                        Topic Distribution
                    </h3>

                    <div className="space-y-4">
                        {topicsData.map(([topic, count]: [string, any], index) => {
                            const maxCount = Math.max(...topicsData.map(([_, c]: any) => c));
                            const percentage = (count / maxCount) * 100;

                            return (
                                <div key={topic} className="space-y-2">
                                    <div className="flex items-center justify-between text-sm">
                                        <span className="font-medium text-text-primary capitalize">{topic}</span>
                                        <span className="text-text-tertiary">{count} questions</span>
                                    </div>

                                    <div className="h-2 bg-bg-tertiary rounded-full overflow-hidden">
                                        <motion.div
                                            initial={{ width: 0 }}
                                            animate={{ width: `${percentage}%` }}
                                            transition={{ delay: index * 0.1, duration: 0.6 }}
                                            className="h-full bg-gradient-primary rounded-full"
                                        />
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </motion.div>
            )}

            {/* Learning Style Info */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="glass rounded-2xl p-6 border border-white/5"
            >
                <h4 className="font-semibold text-text-primary mb-3 flex items-center gap-2">
                    <Zap size={18} className="text-secondary" />
                    Your Learning Style: <span className="gradient-text capitalize">{learningProfile.style || 'Balanced'}</span>
                </h4>

                <p className="text-sm text-text-secondary mb-4">
                    {learningProfile.style === 'detailed'
                        ? 'You prefer comprehensive, step-by-step explanations with in-depth coverage.'
                        : learningProfile.style === 'concise'
                            ? 'You learn best with clear, concise answers that get straight to the point.'
                            : learningProfile.style === 'visual'
                                ? 'You benefit from analogies, mental images, and descriptive language.'
                                : learningProfile.style === 'example-driven'
                                    ? 'You learn effectively through practical examples and demonstrations.'
                                    : 'You have a balanced learning approach that adapts to the topic.'}
                </p>

                <div className="text-xs text-text-tertiary">
                    💡 The AI adapts its responses based on your learning style for optimal understanding
                </div>
            </motion.div>
        </div>
    );
}

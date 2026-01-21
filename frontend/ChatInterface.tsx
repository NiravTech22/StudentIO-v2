import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, User, Sparkles } from 'lucide-react';
import { API_BASE_URL } from '../api_client';

interface ChatMessage {
    id: string;
    sender: 'user' | 'system';
    text: string;
    sentiment?: 'positive' | 'negative' | 'neutral';
}

export function ChatInterface({ studentId }: { studentId: number }) {
    const [messages, setMessages] = useState<ChatMessage[]>([
        {
            id: 'init',
            sender: 'system',
            text: `Hello. I am Student Instance ${studentId}. I am ready to learn with you.`,
            sentiment: 'neutral'
        }
    ]);
    const [input, setInput] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    const sendMessage = async () => {
        if (!input.trim()) return;

        const userMsg: ChatMessage = {
            id: Date.now().toString(),
            sender: 'user',
            text: input
        };

        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsTyping(true);

        try {
            const res = await fetch(`${API_BASE_URL}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ studentId, text: userMsg.text })
            });

            const data = await res.json();

            const systemMsg: ChatMessage = {
                id: (Date.now() + 1).toString(),
                sender: 'system',
                text: data.response,
                sentiment: data.sentiment
            };

            setMessages(prev => [...prev, systemMsg]);
        } catch (e) {
            console.error(e);
            setMessages(prev => [...prev, {
                id: Date.now().toString(),
                sender: 'system',
                text: "I'm having trouble connecting to my neural core. Please try again.",
                sentiment: 'negative'
            }]);
        } finally {
            setIsTyping(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-slate-900/50 backdrop-blur-md rounded-3xl border border-white/10 overflow-hidden shadow-2xl ring-1 ring-white/5">
            {/* Header */}
            <div className="p-4 border-b border-white/5 bg-white/5 flex items-center gap-3">
                <div className="relative">
                    <div className="absolute inset-0 bg-indigo-500 blur-lg opacity-20 animate-pulse" />
                    <Bot className="text-indigo-400 relative z-10" size={20} />
                </div>
                <div>
                    <h3 className="text-sm font-bold text-slate-200">Neural Interface</h3>
                    <div className="text-[10px] text-indigo-400 font-mono flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                        ONLINE
                    </div>
                </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 font-sans" ref={scrollRef}>
                <AnimatePresence mode="popLayout">
                    {messages.map((msg) => (
                        <motion.div
                            key={msg.id}
                            initial={{ opacity: 0, y: 10, scale: 0.95 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.9 }}
                            transition={{ duration: 0.2 }}
                            className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div
                                className={`
                  max-w-[80%] p-3 rounded-2xl text-sm leading-relaxed relative
                  ${msg.sender === 'user'
                                        ? 'bg-indigo-600 text-white rounded-br-none shadow-lg shadow-indigo-900/20'
                                        : 'bg-white/10 text-slate-200 rounded-bl-none border border-white/5'
                                    }
                `}
                            >
                                {msg.sender === 'system' && (
                                    <Sparkles className="absolute -top-3 -left-2 text-indigo-400 opacity-50" size={12} />
                                )}
                                {msg.text}
                            </div>
                        </motion.div>
                    ))}
                </AnimatePresence>

                {isTyping && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="flex items-center gap-1 ml-2"
                    >
                        <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce [animation-delay:-0.3s]" />
                        <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce [animation-delay:-0.15s]" />
                        <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce" />
                    </motion.div>
                )}
            </div>

            {/* Input */}
            <div className="p-4 bg-white/5 border-t border-white/5">
                <div className="relative group">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                        placeholder="Type your message..."
                        className="w-full bg-slate-950/50 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/50 transition-all font-mono"
                        disabled={isTyping}
                    />
                    <button
                        onClick={sendMessage}
                        disabled={!input.trim() || isTyping}
                        className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 bg-indigo-600 rounded-lg text-white opacity-0 group-focus-within:opacity-100 transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:bg-indigo-500"
                    >
                        <Send size={14} />
                    </button>
                </div>
            </div>
        </div>
    );
}

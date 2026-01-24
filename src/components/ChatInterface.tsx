import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, User2, Sparkles, Loader2, ThumbsUp, ThumbsDown } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    sentiment?: string;
    confidence?: number;
}

interface Props {
    studentId: string;
    apiUrl: string;
}

export default function ChatInterface({ studentId, apiUrl }: Props) {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: '0',
            role: 'assistant',
            content: '👋 Hello! I\'m your AI learning assistant. Ask me anything - from math and science to history and programming. I can also help you with documents you\'ve uploaded!',
            timestamp: new Date(),
            sentiment: 'positive'
        }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLTextAreaElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const sendMessage = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: input,
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch(`${apiUrl}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: input,
                    studentId,
                    conversationHistory: messages.slice(-4).map(m => ({
                        question: m.role === 'user' ? m.content : '',
                        answer: m.role === 'assistant' ? m.content : ''
                    }))
                })
            });

            if (!response.ok) throw new Error('Failed to get response');

            const data = await response.json();

            const assistantMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: data.answer,
                timestamp: new Date(),
                sentiment: data.sentiment,
                confidence: data.confidence
            };

            setMessages(prev => [...prev, assistantMessage]);

        } catch (error) {
            console.error('Chat error:', error);
            setMessages(prev => [...prev, {
                id: Date.now().toString(),
                role: 'assistant',
                content: '❌ Sorry, I encountered an error. Please make sure the backend is running and try again.',
                timestamp: new Date(),
                sentiment: 'negative'
            }]);
        } finally {
            setIsLoading(false);
            inputRef.current?.focus();
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const handleFeedback = async (messageId: string, helpful: boolean) => {
        const message = messages.find(m => m.id === messageId);
        if (!message) return;

        try {
            await fetch(`${apiUrl}/api/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    studentId,
                    question: messages[messages.findIndex(m => m.id === messageId) - 1]?.content || '',
                    answer: message.content,
                    helpful
                })
            });
        } catch (error) {
            console.error('Feedback error:', error);
        }
    };

    return (
        <div className="max-w-5xl mx-auto h-[calc(100vh-12rem)] flex flex-col">
            <div className="glass-strong rounded-3xl shadow-2xl overflow-hidden flex flex-col h-full border border-white/10">

                {/* Messages Area */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    <AnimatePresence mode="popLayout">
                        {messages.map((message, index) => (
                            <motion.div
                                key={message.id}
                                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                                animate={{ opacity: 1, y: 0, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.9 }}
                                transition={{ duration: 0.3 }}
                                className={`flex gap-4 ${message.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
                            >
                                {/* Avatar */}
                                <div className={`
                  flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center
                  ${message.role === 'user'
                                        ? 'bg-gradient-primary'
                                        : 'bg-gradient-to-br from-secondary to-primary'
                                    }
                  shadow-lg
                `}>
                                    {message.role === 'user' ? (
                                        <User2 size={20} className="text-white" />
                                    ) : (
                                        <Sparkles size={20} className="text-white" />
                                    )}
                                </div>

                                {/* Message Content */}
                                <div className={`flex-1 max-w-[80%]`}>
                                    <div className={`
                    p-4 rounded-2xl
                    ${message.role === 'user'
                                            ? 'bg-gradient-primary text-white shadow-md'
                                            : 'glass border border-white/10'
                                        }
                  `}>
                                        {message.role === 'assistant' ? (
                                            <div className="prose prose-invert prose-sm max-w-none">
                                                <ReactMarkdown
                                                    components={{
                                                        code({ node, inline, className, children, ...props }) {
                                                            const match = /language-(\w+)/.exec(className || '');
                                                            return !inline && match ? (
                                                                <SyntaxHighlighter
                                                                    style={vscDarkPlus as any}
                                                                    language={match[1]}
                                                                    PreTag="div"
                                                                    className="rounded-lg my-2"
                                                                    {...props}
                                                                >
                                                                    {String(children).replace(/\n$/, '')}
                                                                </SyntaxHighlighter>
                                                            ) : (
                                                                <code className="bg-bg-tertiary px-1.5 py-0.5 rounded text-primary-light" {...props}>
                                                                    {children}
                                                                </code>
                                                            );
                                                        }
                                                    }}
                                                >
                                                    {message.content}
                                                </ReactMarkdown>
                                            </div>
                                        ) : (
                                            <p className="text-sm leading-relaxed">{message.content}</p>
                                        )}
                                    </div>

                                    {/* Metadata */}
                                    <div className={`flex items-center gap-3 mt-2 px-2 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                        <span className="text-xs text-text-tertiary font-mono">
                                            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                        </span>

                                        {message.role === 'assistant' && message.confidence && (
                                            <span className="text-xs text-text-tertiary flex items-center gap-1">
                                                <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />
                                                {Math.round(message.confidence * 100)}%
                                            </span>
                                        )}

                                        {message.role === 'assistant' && index > 0 && (
                                            <div className="flex gap-1">
                                                <button
                                                    onClick={() => handleFeedback(message.id, true)}
                                                    className="p-1 rounded hover:bg-white/10 transition-colors"
                                                    title="Helpful"
                                                >
                                                    <ThumbsUp size={14} className="text-text-tertiary hover:text-success" />
                                                </button>
                                                <button
                                                    onClick={() => handleFeedback(message.id, false)}
                                                    className="p-1 rounded hover:bg-white/10 transition-colors"
                                                    title="Not helpful"
                                                >
                                                    <ThumbsDown size={14} className="text-text-tertiary hover:text-error" />
                                                </button>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>

                    {/* Loading Indicator */}
                    {isLoading && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="flex gap-4"
                        >
                            <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-secondary to-primary flex items-center justify-center">
                                <Sparkles size={20} className="text-white" />
                            </div>
                            <div className="glass border border-white/10 rounded-2xl p-4 flex items-center gap-2">
                                <Loader2 className="animate-spin text-primary" size={18} />
                                <span className="text-sm text-text-secondary">Thinking...</span>
                            </div>
                        </motion.div>
                    )}

                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="p-4 border-t border-white/10 bg-bg-secondary/50 backdrop-blur-sm">
                    <div className="relative group">
                        <textarea
                            ref={inputRef}
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Ask me anything... (Shift+Enter for new line)"
                            className="w-full bg-bg-tertiary border border-white/10 rounded-xl px-4 py-3 pr-14 text-sm text-text-primary placeholder-text-tertiary resize-none focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all font-mono"
                            rows={1}
                            style={{ minHeight: '50px', maxHeight: '150px' }}
                            disabled={isLoading}
                        />

                        <button
                            onClick={sendMessage}
                            disabled={!input.trim() || isLoading}
                            className="absolute right-2 top-1/2 -translate-y-1/2 p-2.5 bg-gradient-primary rounded-lg text-white transition-all hover:shadow-glow-primary-sm disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none"
                        >
                            <Send size={18} />
                        </button>
                    </div>

                    <div className="mt-2 flex items-center gap-2 text-xs text-text-tertiary">
                        <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />
                        <span>Powered by Hugging Face Transformers • Press Enter to send</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

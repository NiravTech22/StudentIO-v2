import { useState } from 'react';
import { Send, Sparkles } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';

const API_URL = 'http://localhost:8000';

interface Message {
    role: 'user' | 'assistant';
    content: string;
}

export default function App() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const sendMessage = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage: Message = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch(`${API_URL}/api/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: input })
            });

            const data = await response.json();
            const assistantMessage: Message = {
                role: 'assistant',
                content: data.answer || 'Sorry, I encountered an error.'
            };

            setMessages(prev => [...prev, assistantMessage]);
        } catch (error) {
            console.error('Error:', error);
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'Failed to connect to the AI. Make sure the Python server is running on port 8000.'
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    return (
        <div className="min-h-screen bg-white flex flex-col">
            {/* Header */}
            <header className="border-b border-gray-200 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
                <div className="max-w-3xl mx-auto px-4 py-4 flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-sm">
                        <Sparkles size={18} className="text-white" />
                    </div>
                    <span className="font-semibold text-gray-900">StudentIO</span>
                </div>
            </header>

            {/* Messages */}
            <main className="flex-1 overflow-y-auto">
                <div className="max-w-3xl mx-auto px-4 py-8">
                    {messages.length === 0 ? (
                        <div className="text-center py-20">
                            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center mx-auto mb-6 shadow-lg">
                                <Sparkles size={32} className="text-white" />
                            </div>
                            <h1 className="text-3xl font-bold text-gray-900 mb-3">
                                How can I help you learn today?
                            </h1>
                            <p className="text-gray-600">
                                Ask me anything about math, science, programming, or any academic topic
                            </p>
                        </div>
                    ) : (
                        <div className="space-y-6">
                            {messages.map((message, index) => (
                                <div
                                    key={index}
                                    className={`flex gap-4 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                                >
                                    {message.role === 'assistant' && (
                                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center flex-shrink-0 mt-1">
                                            <Sparkles size={16} className="text-white" />
                                        </div>
                                    )}

                                    <div
                                        className={`max-w-[80%] rounded-2xl px-5 py-3 ${message.role === 'user'
                                            ? 'bg-indigo-600 text-white'
                                            : 'bg-gray-100 text-gray-900'
                                            }`}
                                    >
                                        {message.role === 'assistant' ? (
                                            <div className="prose prose-sm max-w-none prose-gray">
                                                <ReactMarkdown
                                                    components={{
                                                        code({ node, inline, className, children, style, ...props }: any) {
                                                            const match = /language-(\w+)/.exec(className || '');
                                                            return !inline && match ? (
                                                                <SyntaxHighlighter
                                                                    style={oneLight as any}
                                                                    language={match[1]}
                                                                    PreTag="div"
                                                                    className="rounded-lg my-2 text-sm"
                                                                    {...props}
                                                                >
                                                                    {String(children).replace(/\n$/, '')}
                                                                </SyntaxHighlighter>
                                                            ) : (
                                                                <code className="bg-gray-200 px-1.5 py-0.5 rounded text-indigo-700 font-mono text-xs" {...props}>
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

                                    {message.role === 'user' && (
                                        <div className="w-8 h-8 rounded-lg bg-gray-900 flex items-center justify-center flex-shrink-0 mt-1">
                                            <span className="text-white text-sm font-semibold">You</span>
                                        </div>
                                    )}
                                </div>
                            ))}

                            {isLoading && (
                                <div className="flex gap-4">
                                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center flex-shrink-0">
                                        <Sparkles size={16} className="text-white" />
                                    </div>
                                    <div className="bg-gray-100 rounded-2xl px-5 py-3">
                                        <div className="flex gap-1">
                                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </main>

            {/* Input */}
            <div className="border-t border-gray-200 bg-white sticky bottom-0">
                <div className="max-w-3xl mx-auto px-4 py-4">
                    <div className="relative flex items-center bg-gray-100 rounded-full shadow-sm hover:shadow-md transition-shadow">
                        <textarea
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Ask me anything..."
                            className="flex-1 bg-transparent px-6 py-4 outline-none resize-none text-gray-900 placeholder-gray-500"
                            rows={1}
                            style={{ maxHeight: '120px' }}
                            disabled={isLoading}
                        />
                        <button
                            onClick={sendMessage}
                            disabled={!input.trim() || isLoading}
                            className="mr-2 p-3 rounded-full bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-all hover:scale-105 active:scale-95"
                        >
                            <Send size={18} />
                        </button>
                    </div>
                    <p className="text-center text-xs text-gray-500 mt-3">
                        Powered by AI • Press Enter to send
                    </p>
                </div>
            </div>
        </div>
    );
}

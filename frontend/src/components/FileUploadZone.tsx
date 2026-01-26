import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, File, FileText, Image as ImageIcon, Presentation, Loader2, CheckCircle2, XCircle, Trash2 } from 'lucide-react';

interface UploadedFile {
    filename: string;
    size: number;
    type: string;
    status: 'uploading' | 'processed' | 'error';
    chunks?: number;
    error?: string;
}

interface Props {
    studentId: string;
    apiUrl: string;
}

export default function FileUploadZone({ studentId, apiUrl }: Props) {
    const [files, setFiles] = useState<UploadedFile[]>([]);
    const [isDragging, setIsDragging] = useState(false);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    }, []);

    const uploadFiles = async (fileList: FileList) => {
        const newFiles: UploadedFile[] = Array.from(fileList).map(f => ({
            filename: f.name,
            size: f.size,
            type: f.type,
            status: 'uploading'
        }));

        setFiles(prev => [...prev, ...newFiles]);
        setFiles(prev => [...prev, ...newFiles]);

        const formData = new FormData();
        Array.from(fileList).forEach(file => {
            formData.append('files', file);
        });
        formData.append('studentId', studentId);

        try {
            const response = await fetch(`${apiUrl}/api/files/upload`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            setFiles(prev => prev.map(file => {
                const result = data.files?.find((f: any) => f.filename === file.filename);
                if (result) {
                    return {
                        ...file,
                        status: result.status as any,
                        chunks: result.chunks,
                        error: result.error
                    };
                }
                return file;
            }));

        } catch (error) {
            console.error('Upload error:', error);
            setFiles(prev => prev.map(f =>
                newFiles.find(nf => nf.filename === f.filename)
                    ? { ...f, status: 'error', error: 'Upload failed' }
                    : f
            ));
        }
    };

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        if (e.dataTransfer.files) {
            uploadFiles(e.dataTransfer.files);
        }
    }, [studentId, apiUrl]);

    const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            uploadFiles(e.target.files);
        }
    };

    const removeFile = (filename: string) => {
        setFiles(prev => prev.filter(f => f.filename !== filename));
    };

    const getFileIcon = (type: string) => {
        if (type.includes('pdf')) return FileText;
        if (type.includes('word') || type.includes('document')) return File;
        if (type.includes('image')) return ImageIcon;
        if (type.includes('presentation')) return Presentation;
        return File;
    };

    const formatFileSize = (bytes: number) => {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    };

    return (
        <div className="max-w-5xl mx-auto space-y-6">
            {/* Upload Zone */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className={`
          relative glass-strong border-2 border-dashed rounded-3xl p-12 text-center
          transition-all duration-300 cursor-pointer group overflow-hidden
          ${isDragging
                        ? 'border-primary bg-primary/10 scale-[1.02]'
                        : 'border-white/20 hover:border-primary/50 hover:bg-white/5'
                    }
        `}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-input')?.click()}
            >
                {/* Animated Background */}
                <div className="absolute inset-0 bg-gradient-primary opacity-0 group-hover:opacity-10 transition-opacity duration-500" />

                <div className="relative z-10">
                    <Upload
                        size={64}
                        className={`mx-auto mb-4 transition-all duration-300 ${isDragging ? 'text-primary scale-110' : 'text-text-secondary group-hover:text-primary'
                            }`}
                    />

                    <h3 className="text-2xl font-bold mb-2 text-text-primary">
                        {isDragging ? 'Drop files here' : 'Upload Learning Materials'}
                    </h3>

                    <p className="text-text-secondary mb-6">
                        Drag & drop or click to upload PDF, DOCX, images, or slides
                    </p>

                    <div className="flex flex-wrap justify-center gap-2 text-xs text-text-tertiary">
                        {['PDF', 'DOCX', 'PNG', 'JPG', 'PPTX'].map(format => (
                            <span key={format} className="px-3 py-1 rounded-full bg-white/5 border border-white/10">
                                {format}
                            </span>
                        ))}
                    </div>
                </div>

                <input
                    id="file-input"
                    type="file"
                    multiple
                    accept=".pdf,.docx,.doc,.pptx,.ppt,.png,.jpg,.jpeg,.tiff,.bmp"
                    onChange={handleFileInput}
                    className="hidden"
                />
            </motion.div>

            {/* Uploaded Files List */}
            {files.length > 0 && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass-strong rounded-3xl p-6 border border-white/10"
                >
                    <h3 className="text-lg font-bold mb-4 text-text-primary flex items-center gap-2">
                        <File size={20} className="text-primary" />
                        Uploaded Files ({files.length})
                    </h3>

                    <div className="space-y-3">
                        <AnimatePresence mode="popLayout">
                            {files.map((file, index) => {
                                const Icon = getFileIcon(file.type);

                                return (
                                    <motion.div
                                        key={file.filename}
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        exit={{ opacity: 0, x: 20 }}
                                        transition={{ delay: index * 0.05 }}
                                        className="flex items-center gap-4 p-4 rounded-xl glass border border-white/5 hover:border-primary/30 transition-all group"
                                    >
                                        {/* Icon */}
                                        <div className={`
                      p-2 rounded-lg flex-shrink-0
                      ${file.status === 'processed' ? 'bg-success/20' : ''}
                      ${file.status === 'error' ? 'bg-error/20' : ''}
                      ${file.status === 'uploading' ? 'bg-primary/20' : ''}
                    `}>
                                            <Icon size={24} className={`
                        ${file.status === 'processed' ? 'text-success' : ''}
                        ${file.status === 'error' ? 'text-error' : ''}
                        ${file.status === 'uploading' ? 'text-primary' : ''}
                      `} />
                                        </div>

                                        {/* File Info */}
                                        <div className="flex-1 min-w-0">
                                            <p className="text-sm font-medium text-text-primary truncate">
                                                {file.filename}
                                            </p>
                                            <div className="flex items-center gap-3 mt-1">
                                                <span className="text-xs text-text-tertiary">
                                                    {formatFileSize(file.size)}
                                                </span>
                                                {file.chunks !== undefined && (
                                                    <span className="text-xs text-success">
                                                        {file.chunks} chunks processed
                                                    </span>
                                                )}
                                                {file.error && (
                                                    <span className="text-xs text-error">
                                                        {file.error}
                                                    </span>
                                                )}
                                            </div>
                                        </div>

                                        {/* Status Indicator */}
                                        <div className="flex items-center gap-2">
                                            {file.status === 'uploading' && (
                                                <Loader2 className="animate-spin text-primary" size={20} />
                                            )}
                                            {file.status === 'processed' && (
                                                <CheckCircle2 className="text-success" size={20} />
                                            )}
                                            {file.status === 'error' && (
                                                <XCircle className="text-error" size={20} />
                                            )}

                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    removeFile(file.filename);
                                                }}
                                                className="p-1.5 rounded-lg hover:bg-error/20 transition-colors opacity-0 group-hover:opacity-100"
                                            >
                                                <Trash2 size={16} className="text-error" />
                                            </button>
                                        </div>
                                    </motion.div>
                                );
                            })}
                        </AnimatePresence>
                    </div>
                </motion.div>
            )}

            {/* Info Card */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="glass rounded-2xl p-6 border border-white/5"
            >
                <h4 className="font-semibold text-text-primary mb-3">How it works</h4>
                <ul className="space-y-2 text-sm text-text-secondary">
                    <li className="flex items-start gap-2">
                        <CheckCircle2 size={16} className="text-success mt-0.5 flex-shrink-0" />
                        <span>Upload your study materials (lectures, notes, slides)</span>
                    </li>
                    <li className="flex items-start gap-2">
                        <CheckCircle2 size={16} className="text-success mt-0.5 flex-shrink-0" />
                        <span>AI automatically extracts and indexes the content</span>
                    </li>
                    <li className="flex items-start gap-2">
                        <CheckCircle2 size={16} className="text-success mt-0.5 flex-shrink-0" />
                        <span>Ask questions and get answers based on YOUR documents</span>
                    </li>
                </ul>
            </motion.div>
        </div>
    );
}

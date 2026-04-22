/**
 * Upload Page - Document Upload Interface
 * 
 * Drag & drop or click to upload:
 * - PDF files
 * - DOCX files
 * - TXT files
 */

'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import { 
  FileText, 
  AlertTriangle,
  Loader2,
  Sparkles
} from 'lucide-react';
import { toast } from 'sonner';
import { FileUploadCard, UploadedFile } from '@/components/ui/file-upload-card';
import { Input } from '@/components/ui/input';

export default function UploadPage() {
  const router = useRouter();
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [documentId, setDocumentId] = useState('');
  const [jurisdiction, setJurisdiction] = useState('');
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warming, setWarming] = useState(true);

  // Warm up backend on page load so first upload is faster
  useEffect(() => {
    api.getDocuments()
      .catch(() => {}) // silently ignore errors — just waking the server
      .finally(() => setWarming(false));
  }, []);

  const handleFilesChange = (newFiles: File[]) => {
    // Validate each file
    const validTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
    const validExtensions = ['.pdf', '.docx', '.txt'];
    
    const validatedFiles: UploadedFile[] = [];
    
    for (const file of newFiles) {
      const isValidType = validTypes.includes(file.type) || 
                          validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
      
      if (!isValidType) {
        toast.error('Invalid file type', {
          description: `${file.name}: Please upload PDF, DOCX, or TXT files only`
        });
        continue;
      }

      if (file.size > 10 * 1024 * 1024) {
        toast.error('File too large', {
          description: `${file.name}: Maximum file size is 10MB`
        });
        continue;
      }

      validatedFiles.push({
        id: `${file.name}-${Date.now()}-${Math.random()}`,
        file,
        progress: 0,
        status: 'uploading'
      });
    }

    if (validatedFiles.length > 0) {
      setFiles(prev => [...prev, ...validatedFiles]);
      setError(null);

      // Auto-generate document ID from first file if not set
      if (!documentId && validatedFiles[0]) {
        const autoId = validatedFiles[0].file.name
          .replace(/\.[^/.]+$/, '')
          .replace(/[^a-zA-Z0-9_-]/g, '_');
        setDocumentId(autoId);
      }

      toast.success('File selected', {
        description: `${validatedFiles.length} file(s) ready to upload`
      });
    }
  };

  const handleFileRemove = (id: string) => {
    setFiles(prev => prev.filter(f => f.id !== id));
  };

  // Simulate upload progress for files in "uploading" status
  useEffect(() => {
    const uploadingFiles = files.filter(f => f.status === 'uploading');
    if (uploadingFiles.length === 0) return;

    const interval = setInterval(() => {
      setFiles(prevFiles =>
        prevFiles.map(f => {
          if (f.status === 'uploading') {
            const newProgress = Math.min(f.progress + 5, 100);
            return {
              ...f,
              progress: newProgress,
              status: newProgress === 100 ? 'completed' : 'uploading'
            };
          }
          return f;
        })
      );
    }, 100);

    return () => clearInterval(interval);
  }, [files]);

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();

    const completedFiles = files.filter(f => f.status === 'completed');
    if (completedFiles.length === 0) {
      setError('Please select a file and wait for it to be ready');
      toast.error('No file ready', {
        description: 'Please select a file to upload'
      });
      return;
    }

    // For now, upload the first completed file (maintaining single-file behavior)
    const fileToUpload = completedFiles[0];
    setUploading(true);
    setError(null);

    const uploadPromise = api.uploadDocument(
      fileToUpload.file,
      documentId || undefined,
      jurisdiction || undefined
    );

    toast.promise(uploadPromise, {
      loading: 'Processing document...',
      success: (result) => {
        // Reset form
        setFiles([]);
        setDocumentId('');
        setJurisdiction('');

        // Redirect to documents page after brief delay
        setTimeout(() => {
          router.push('/documents');
        }, 1500);

        return `Successfully uploaded "${result.filename}" with ${result.chunks_created} chunks`;
      },
      error: (err: any) => {
        const errorMsg = err.response?.data?.detail || 'Failed to upload document';
        setError(errorMsg);
        return errorMsg;
      },
    });

    try {
      await uploadPromise;
    } catch (err) {
      console.error('Upload failed:', err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="min-h-screen px-8 py-12">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12 animate-fade-in">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass border border-white/10 mb-6">
            <Sparkles className="w-4 h-4 text-blue-400" />
            <span className="text-sm font-medium text-secondary">
              Document Ingestion Pipeline
            </span>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-3">Upload Document</h1>
          <p className="text-xl text-secondary max-w-2xl mx-auto">
            Add legal documents to your knowledge base with AI-powered processing
          </p>
        </div>

        <form onSubmit={handleUpload} className="space-y-6 animate-fade-in-up">
          {/* File Upload Card */}
          <FileUploadCard
            files={files}
            onFilesChange={handleFilesChange}
            onFileRemove={handleFileRemove}
            className="w-full"
          />

          {/* Metadata Inputs */}
          <div className="bg-surface border border-white/10 rounded-2xl p-6 space-y-4">
            <h3 className="text-lg font-semibold flex items-center gap-2 mb-4">
              <FileText className="w-5 h-5 text-blue-400" />
              Document Metadata
            </h3>

            <div>
              <label className="block text-sm font-medium mb-2 text-secondary">
                Document ID (Optional)
              </label>
              <Input
                type="text"
                value={documentId}
                onChange={(e) => setDocumentId(e.target.value)}
                placeholder="Auto-generated from filename"
                className="h-11 rounded-xl bg-background border-white/10 focus-visible:ring-blue-500/30 focus-visible:ring-2"
                disabled={uploading}
              />
              <p className="text-xs text-muted mt-2">
                Unique identifier for this document (e.g., EMP_AGREE_001)
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2 text-secondary">
                Jurisdiction (Optional)
              </label>
              <Input
                type="text"
                value={jurisdiction}
                onChange={(e) => setJurisdiction(e.target.value)}
                placeholder="E.g., California, New York, Federal"
                className="h-11 rounded-xl bg-background border-white/10 focus-visible:ring-blue-500/30 focus-visible:ring-2"
                disabled={uploading}
              />
              <p className="text-xs text-muted mt-2">
                Legal jurisdiction this document applies to
              </p>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="p-6 rounded-2xl bg-rose-500/10 border border-rose-500/20 animate-fade-in">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-rose-400 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-rose-400 font-medium">{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={files.filter(f => f.status === 'completed').length === 0 || uploading || warming}
            className="w-full h-12 rounded-xl bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center justify-center gap-2"
          >
            {warming ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Connecting to backend...
              </>
            ) : uploading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Processing Document... (may take up to 60s)
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                Upload & Process
              </>
            )}
          </button>
        </form>

        {/* Processing Steps */}
        <div className="mt-12 p-6 rounded-2xl bg-surface border border-white/10 animate-fade-in-up delay-200">
          <h4 className="font-semibold text-blue-400 mb-5 flex items-center gap-2">
            <Sparkles className="w-4 h-4" />
            Processing Pipeline
          </h4>
          <div className="space-y-0">
            <ProcessStep number={1} text="Document parsing (PDF/DOCX/TXT)" isLast={false} />
            <ProcessStep number={2} text="Clause-aware semantic chunking" isLast={false} />
            <ProcessStep number={3} text="Embedding generation (sentence-transformers)" isLast={false} />
            <ProcessStep number={4} text="Vector indexing (pgvector / Supabase)" isLast={false} />
            <ProcessStep number={5} text="Ready for semantic search!" isLast />
          </div>
        </div>
      </div>
    </div>
  );
}

function ProcessStep({
  number,
  text,
  isLast,
}: {
  number: number;
  text: string;
  isLast: boolean;
}) {
  return (
    <div className="relative flex items-start gap-3 py-2.5">
      <div className="relative flex h-6 w-6 items-center justify-center flex-shrink-0">
        {!isLast && (
          <span className="absolute top-6 left-1/2 h-[calc(100%+8px)] w-px -translate-x-1/2 bg-white/10" />
        )}
        <div className="h-6 w-6 rounded-full bg-blue-500/15 border border-blue-400/30 flex items-center justify-center">
          <span className="text-[11px] font-semibold text-blue-300">{number}</span>
        </div>
      </div>
      <div className="min-h-6 flex items-center">
        <span className="text-sm text-secondary">{text}</span>
      </div>
    </div>
  );
}

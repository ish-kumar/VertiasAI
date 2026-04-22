/**
 * API Client - Frontend to Backend Communication
 * 
 * All API calls to the FastAPI backend go through this module.
 */

import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 120000, // 120 seconds (cold start + LLM calls on free tier)
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface QueryRequest {
  query: string;
  jurisdiction?: string;
  context?: string;
}

export interface QueryResponse {
  success: boolean;
  query_id: string;
  query: string;
  answer?: {
    answer_text: string;
    citations: Array<{
      clause_id: string;
      quoted_text: string;
      reasoning: string;
    }>;
    reasoning: string;
    assumptions: string[];
    caveats: string[];
  };
  counter_arguments?: {
    contradictions?: string[];
    exceptions?: string[];
    jurisdictional_issues?: string[];
    ambiguities?: string[];
    missing_context?: string[];
    alternative_interpretation?: string;
    severity: string;
  };
  verification?: {
    valid_citations: number;
    invalid_citations: number;
  };
  confidence?: {
    overall_score: number;
    citation_score: number;
    counter_strength: number;
  };
  risk?: {
    overall_risk: string;
    factors: string[];
  };
  warnings?: string[];
  refusal_explanation?: string;
}

export interface DocumentInfo {
  document_id: string;
  chunk_count: number;
  jurisdiction: string;
  doc_type: string;
}

export interface SystemStats {
  total_documents: number;
  total_chunks: number;
  embedding_model: string;
  embedding_dimension: number;
  memory_usage_mb: number;
  status: string;
}

export const api = {
  /**
   * Submit a legal query to the RAG pipeline.
   */
  submitQuery: async (request: QueryRequest): Promise<QueryResponse> => {
    const response = await apiClient.post<QueryResponse>('/query', request);
    return response.data;
  },

  /**
   * Upload and ingest a document.
   */
  uploadDocument: async (
    file: File,
    documentId?: string,
    jurisdiction?: string
  ): Promise<any> => {
    const formData = new FormData();
    formData.append('file', file);
    if (documentId) formData.append('document_id', documentId);
    if (jurisdiction) formData.append('jurisdiction', jurisdiction);

    const response = await apiClient.post('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  /**
   * Get list of all indexed documents.
   */
  getDocuments: async (): Promise<{ total_documents: number; documents: DocumentInfo[] }> => {
    const response = await apiClient.get('/documents');
    return response.data;
  },

  /**
   * Delete a document from the index.
   */
  deleteDocument: async (documentId: string): Promise<any> => {
    const response = await apiClient.delete(`/documents/${documentId}`);
    return response.data;
  },

  /**
   * Get system statistics.
   */
  getStats: async (): Promise<SystemStats> => {
    const response = await apiClient.get<SystemStats>('/stats');
    return response.data;
  },

  /**
   * Health check.
   */
  healthCheck: async (): Promise<{ status: string; service: string }> => {
    const response = await apiClient.get('/health');
    return response.data;
  },
};

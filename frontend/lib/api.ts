import axios from "axios";
import type { TranslateResponse, HealthResponse, ErrorResponse } from "@/types";

// Normalize API URL by removing trailing slashes
const normalizeUrl = (url: string): string => {
  return url.replace(/\/+$/, '');
};

const API_BASE_URL = normalizeUrl(process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001");

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds
  headers: {
    "Content-Type": "application/json",
  },
});

// Add request interceptor for logging and auth token
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);

    // Add auth token if available
    if (typeof window !== "undefined") {
      const token = localStorage.getItem("auth_token");
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }

    return config;
  },
  (error) => {
    return Promise.reject(error);
  },
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error("API Error:", error.response?.data || error.message);

    // Handle invalid/expired token
    if (error.response?.status === 422 || error.response?.status === 401) {
      const errorMsg =
        error.response?.data?.msg || error.response?.data?.error || "";

      // Check if it's a JWT token error
      if (
        errorMsg.includes("Subject must be a string") ||
        errorMsg.includes("token") ||
        errorMsg.includes("expired") ||
        errorMsg.includes("invalid")
      ) {
        console.warn(
          "Invalid/expired token detected. Clearing localStorage...",
        );

        // Clear auth data
        if (typeof window !== "undefined") {
          localStorage.removeItem("auth_token");
          localStorage.removeItem("user");

          // Show alert and reload
          alert("Your session has expired. Please login again.");
          window.location.reload();
        }
      }
    }

    return Promise.reject(error);
  },
);

export const apiClient = {
  /**
   * Check API health status
   */
  async checkHealth(): Promise<HealthResponse> {
    const response = await api.get<HealthResponse>("/health");
    return response.data;
  },

  /**
   * Check model loading status
   */
  async checkModelsStatus(): Promise<any> {
    const response = await api.get("/models/status");
    return response.data;
  },

  /**
   * Upload and translate image
   */
  async translateImage(file: File): Promise<TranslateResponse> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await api.post<TranslateResponse>(
      "/translate",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      },
    );

    return response.data;
  },

  /**
   * OCR only (no translation)
   */
  async ocrOnly(file: File): Promise<any> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await api.post("/ocr-only", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    return response.data;
  },

  /**
   * Translate text directly (no OCR)
   */
  async translateText(text: string): Promise<any> {
    const response = await api.post("/translate-text", { text });
    return response.data;
  },

  /**
   * Aksara Sunda Handwritten OCR only (no translation)
   */
  async aksaraOcrOnly(file: File): Promise<any> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await api.post("/aksara-ocr", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    return response.data;
  },

  /**
   * Aksara Sunda Unicode OCR only (no translation)
   */
  async aksaraUnicodeOcrOnly(file: File): Promise<any> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await api.post("/aksara-unicode-ocr", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    return response.data;
  },

  /**
   * Aksara Sunda Handwritten OCR + Translation
   * For scanned handwritten documents
   */
  async translateAksaraHandwritten(file: File): Promise<TranslateResponse> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await api.post<TranslateResponse>(
      "/translate-aksara",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      },
    );

    return response.data;
  },

  /**
   * Aksara Sunda Unicode OCR + Translation
   * For digital screenshots and PDFs
   */
  async translateAksaraUnicode(file: File): Promise<TranslateResponse> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await api.post<TranslateResponse>(
      "/translate-aksara-unicode",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      },
    );

    return response.data;
  },

  /**
   * Upload audio file for transcription and translation
   */
  async uploadAudio(file: File): Promise<any> {
    const formData = new FormData();
    formData.append("audio", file);

    const response = await api.post("/api/upload-audio", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    return response.data;
  },

  /**
   * Upload video file for audio-visual transcription and translation
   */
  async uploadVideo(file: File): Promise<any> {
    const formData = new FormData();
    formData.append("video", file);

    const response = await api.post("/api/upload-video", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    return response.data;
  },

  /**
   * Get translation history
   */
  async getTranslationHistory(page: number = 1, perPage: number = 10): Promise<any> {
    const response = await api.get(`/api/translation-history?page=${page}&per_page=${perPage}`);
    return response.data;
  },

  /**
   * Delete translation history entry
   */
  async deleteTranslationHistory(historyId: number): Promise<any> {
    const response = await api.delete(`/api/translation-history/${historyId}`);
    return response.data;
  },
};

export default api;

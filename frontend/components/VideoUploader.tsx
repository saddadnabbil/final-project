// frontend/components/VideoUploader.tsx
"use client";

import { useState, useRef, useCallback } from "react";

export default function VideoUploader() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcription, setTranscription] = useState("");
  const [translation, setTranslation] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('video/')) {
      setError('Please select a video file');
      return;
    }

    // Validate file size (max 100MB)
    if (file.size > 100 * 1024 * 1024) {
      setError('File size must be less than 100MB');
      return;
    }

    setSelectedFile(file);
    setError(null);
    setTranscription("");
    setTranslation("");

    // Create video preview
    const videoUrl = URL.createObjectURL(file);
    setVideoPreview(videoUrl);

    // Load video in preview element
    if (videoRef.current) {
      videoRef.current.src = videoUrl;
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      const file = files[0];
      handleFileSelect({ target: { files: [file] } } as any);
    }
  }, [handleFileSelect]);

  const handleUpload = useCallback(async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('video', selectedFile);

      const response = await fetch('http://localhost:5001/api/upload-video', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();

      if (result.success) {
        setTranscription(result.transcription || '');
        setTranslation(result.translation || '');
      } else {
        setError(result.error || 'Processing failed');
      }
    } catch (err) {
      console.error('Upload error:', err);
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsProcessing(false);
    }
  }, [selectedFile]);

  const handleClear = useCallback(() => {
    setSelectedFile(null);
    setVideoPreview(null);
    setTranscription("");
    setTranslation("");
    setError(null);
    if (videoRef.current) {
      videoRef.current.src = '';
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  return (
    <div className="p-4 w-full  mx-auto rounded-xl flex flex-col">
      <div className="text-center mb-6">
        <h1 className="text-3xl font-bold mb-2 text-gray-800">Penerjemah Video Sunda (Audio-Visual)</h1>
        <p className="text-gray-600">Upload video untuk transkripsi dan terjemahan otomatis</p>
      </div>

      {error && (
        <div className="my-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded-lg text-center">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* File Upload Section */}
      <div className="my-4">
        <div
          className={`border-2 border-dashed rounded-lg p-6 text-center transition-all duration-200 ${
            isDragOver
              ? 'border-blue-500 bg-blue-50 scale-105'
              : 'border-gray-300 hover:border-blue-400'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileSelect}
            className="hidden"
            id="video-upload"
          />
          <label htmlFor="video-upload" className="cursor-pointer block">
            <div className={`mb-2 transition-colors ${isDragOver ? 'text-blue-600' : 'text-gray-600'}`}>
              <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <p className={`text-lg font-medium mb-1 transition-colors ${
              isDragOver ? 'text-blue-700' : 'text-gray-900'
            }`}>
              {isDragOver
                ? 'Lepaskan file video di sini'
                : selectedFile
                  ? selectedFile.name
                  : 'Klik untuk upload video atau tarik & lepaskan'
              }
            </p>
            <p className="text-sm text-gray-500">
              MP4, WebM, AVI hingga 100MB
            </p>
            {isDragOver && (
              <div className="mt-2 text-blue-600 font-medium animate-pulse">
                📹 Ready to accept video file
              </div>
            )}
          </label>
        </div>
      </div>

      {/* Video Preview */}
      {videoPreview && (
        <div className="my-4 rounded-xl overflow-hidden flex justify-center">
          <video
            ref={videoRef}
            controls
            className="w-full max-w-lg aspect-video h-auto rounded-xl bg-black"
            style={{ minHeight: 220, maxHeight: 400 }}
          />
        </div>
      )}

      {/* Action Buttons */}
      {selectedFile && (
        <div className="flex justify-center gap-4 my-6">
          <button
            onClick={handleUpload}
            disabled={isProcessing}
            className="px-6 py-2 bg-blue-600 text-white font-semibold rounded shadow hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed transition-colors text-base"
          >
            {isProcessing ? 'Memproses...' : 'Upload & Proses'}
          </button>
          <button
            onClick={handleClear}
            disabled={isProcessing}
            className="px-6 py-2 bg-gray-600 text-white font-semibold rounded shadow hover:bg-gray-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors text-base"
          >
            Clear
          </button>
        </div>
      )}

      {/* Processing Status */}
      {isProcessing && (
        <div className="my-4 p-4 border border-yellow-300 bg-yellow-50 rounded text-yellow-800 text-center animate-pulse font-medium">
          <span className="inline-flex items-center gap-2">
            <svg width="18" height="18" fill="none" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="10" fill="#fefcbf"/>
              <path d="M12 8v4m0 4h.01" stroke="#d69e2e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="12" cy="12" r="9" stroke="#d69e2e" strokeWidth="2"/>
            </svg>
            Sedang memproses video & menerjemahkan...
          </span>
        </div>
      )}

      {/* Results */}
      {(transcription || translation) && (
        <div className="mt-4">
          <h2 className="font-bold text-xl mb-2 text-gray-800">Hasil Transkripsi & Terjemahan:</h2>

          {/* Transcription (Sunda) */}
          {transcription && (
            <div className="mb-3">
              <h3 className="font-semibold text-gray-700 mb-1 text-sm">Transkripsi (Sunda):</h3>
              <div className="p-3 border border-blue-200 bg-blue-50 rounded text-gray-900 whitespace-pre-wrap font-normal text-base">
                {transcription}
              </div>
            </div>
          )}

          {/* Translation (Indonesia) */}
          <div className="mb-2">
            <h3 className="font-semibold text-gray-700 mb-1 text-sm">Terjemahan (Indonesia):</h3>
            <div className="p-3 border border-green-200 bg-green-50 rounded min-h-[80px] text-gray-900 whitespace-pre-wrap font-normal text-base">
              {translation || <span className="text-gray-400">Upload video untuk mendapatkan hasil transkripsi & terjemahan.</span>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

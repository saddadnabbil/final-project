"use client";

import { useCallback } from "react";
import { Upload } from "lucide-react";

interface AudioUploadProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
  acceptVideo?: boolean;
}

export default function AudioUpload({
  onFileSelect,
  disabled = false,
  acceptVideo = false,
}: AudioUploadProps) {
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();

      const file = e.dataTransfer.files?.[0];
      if (file) {
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const acceptTypes = acceptVideo
    ? "audio/*,video/*"
    : "audio/*";

  const fileTypeText = acceptVideo
    ? "audio or video"
    : "audio";

  return (
    <div
      className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-8 text-center hover:border-primary/50 transition-colors"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      <input
        type="file"
        id="audio-upload"
        className="hidden"
        accept={acceptTypes}
        onChange={handleChange}
        disabled={disabled}
      />
      <label
        htmlFor="audio-upload"
        className={`cursor-pointer block ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
        <p className="text-lg font-medium mb-2">
          Choose {fileTypeText} file or drag and drop
        </p>
        <p className="text-sm text-muted-foreground">
          {acceptVideo
            ? "Supports MP3, WAV, MP4, WebM (max 100MB)"
            : "Supports MP3, WAV, M4A (max 16MB)"}
        </p>
      </label>
    </div>
  );
}

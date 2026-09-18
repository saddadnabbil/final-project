"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, Image as ImageIcon } from "lucide-react";
import { validateImageFile } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";
import { Card } from "@/components/ui/card";

interface ImageUploadProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export default function ImageUpload({
  onFileSelect,
  disabled,
}: ImageUploadProps) {
  const { toast } = useToast();

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        const validation = validateImageFile(file);

        if (!validation.valid) {
          toast({
            variant: "destructive",
            title: "Invalid file",
            description: validation.error,
          });
          return;
        }

        onFileSelect(file);
      }
    },
    [onFileSelect, toast],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/jpeg": [".jpg", ".jpeg"],
      "image/png": [".png"],
      "image/webp": [".webp"],
    },
    multiple: false,
    disabled,
  });

  return (
    <Card
      {...getRootProps()}
      className={`
        border-2 border-dashed rounded-lg p-12 text-center cursor-pointer
        transition-all duration-200
        ${
          isDragActive
            ? "border-primary bg-primary/10 shadow-md"
            : "border-muted-foreground/25 hover:border-primary hover:bg-accent/50"
        }
        ${disabled ? "opacity-50 cursor-not-allowed" : ""}
      `}
    >
      <input {...getInputProps()} />
      <div className="flex flex-col items-center justify-center space-y-4">
        {isDragActive ? (
          <>
            <Upload className="w-16 h-16 text-primary animate-bounce" />
            <p className="text-lg font-medium text-primary">
              Drop the image here
            </p>
          </>
        ) : (
          <>
            <ImageIcon className="w-16 h-16 text-muted-foreground" />
            <div>
              <p className="text-lg font-medium text-foreground">
                Drag & drop an image here
              </p>
              <p className="mt-1 text-sm text-muted-foreground">
                or click to select a file
              </p>
            </div>
            <p className="text-xs text-muted-foreground">
              Supports JPG, PNG, WebP (Max 10MB)
            </p>
          </>
        )}
      </div>
    </Card>
  );
}

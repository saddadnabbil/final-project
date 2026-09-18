"use client";

import { X } from "lucide-react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface ImagePreviewProps {
  previewUrl: string;
  onRemove: () => void;
}

export default function ImagePreview({
  previewUrl,
  onRemove,
}: ImagePreviewProps) {
  return (
    <Card className="relative overflow-hidden">
      <div className="relative w-full aspect-video bg-muted">
        <Image src={previewUrl} alt="Preview" fill className="object-contain" />
      </div>
      <Button
        onClick={onRemove}
        size="icon"
        variant="destructive"
        className="absolute top-2 right-2 rounded-full"
        aria-label="Remove image"
      >
        <X className="w-4 h-4" />
      </Button>
    </Card>
  );
}

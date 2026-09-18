"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { FileAudio, Languages, Activity, Clock } from "lucide-react";
import type { AudioVideoResponse } from "@/types";

interface AudioVideoResultProps {
  result: AudioVideoResponse;
}

export default function AudioVideoResult({ result }: AudioVideoResultProps) {
  const formatDuration = (seconds?: number) => {
    if (!seconds) return "N/A";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-4">
      {/* Mode Badge */}
      <div className="flex items-center gap-2">
        <Badge variant={result.mode === "audio-visual" ? "default" : "secondary"}>
          {result.mode === "audio-visual" ? "Audio-Visual Mode" : "Audio Only Mode"}
        </Badge>
      </div>

      {/* Transcription */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <FileAudio className="w-4 h-4" />
            Transcription (Sundanese)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-3 bg-muted rounded-lg">
            <p className="text-sm whitespace-pre-wrap">
              {result.transcription || "No transcription available"}
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Translation */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Languages className="w-4 h-4" />
            Translation (Indonesian)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-3 bg-primary/5 rounded-lg border border-primary/20">
            <p className="text-sm whitespace-pre-wrap">
              {result.translation || "No translation available"}
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Performance Metrics Card - Combined with Visual Augmentation */}
      {result.metrics && (
        <Card className="border-muted">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Performance Metrics
            </CardTitle>
            <CardDescription>Quality and performance statistics</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Time Metrics */}
            <div className="grid grid-cols-2 gap-4">
              {result.metrics.processing_time !== undefined && (
                <div className="space-y-1">
                  <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    <Clock className="w-3 h-3" />
                    <span>Processing Time</span>
                  </div>
                  <p className="text-lg font-semibold">{result.metrics.processing_time.toFixed(2)}s</p>
                </div>
              )}
              {result.metrics.audio_duration !== undefined && (
                <div className="space-y-1">
                  <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    <Clock className="w-3 h-3" />
                    <span>Audio Duration</span>
                  </div>
                  <p className="text-lg font-semibold">{formatDuration(result.metrics.audio_duration)}</p>
                </div>
              )}
            </div>

            {/* Visual Augmentation Section - Inside same card */}
            {result.visual_augmentation && result.mode === "audio-visual" && (
              <div className="pt-3 border-t space-y-3">
                <div className="flex flex-col md:flex-row gap-4">
                  {/* Landmark Image */}
                  {result.visual_augmentation.landmark_image && (
                    <div className="flex-shrink-0">
                      <img
                        src={`data:image/jpeg;base64,${result.visual_augmentation.landmark_image}`}
                        alt="Facial Landmark"
                        className="rounded-lg border w-28 h-auto"
                      />
                    </div>
                  )}

                  {/* Visual Info */}
                  <div className="flex-1 space-y-2">
                    {/* Attention Bar */}
                    <div>
                      <p className="text-xs text-muted-foreground mb-1">Attention Weights</p>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-blue-500">Audio {(result.visual_augmentation.attention_weights.audio * 100).toFixed(0)}%</span>
                        <span className="text-purple-500">Visual {(result.visual_augmentation.attention_weights.visual * 100).toFixed(0)}%</span>
                      </div>
                      <div className="h-2 bg-muted rounded-full overflow-hidden flex">
                        <div
                          className="h-full bg-blue-500"
                          style={{ width: `${result.visual_augmentation.attention_weights.audio * 100}%` }}
                        />
                        <div
                          className="h-full bg-purple-500"
                          style={{ width: `${result.visual_augmentation.attention_weights.visual * 100}%` }}
                        />
                      </div>
                    </div>

                    {/* Face Detection Status */}
                    <Badge variant={result.visual_augmentation.facial_landmarks.detected ? "default" : "secondary"} className="font-normal text-xs">
                      {result.visual_augmentation.facial_landmarks.detected
                        ? `${result.visual_augmentation.facial_landmarks.landmark_count} landmarks detected`
                        : "No face detected"}
                    </Badge>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

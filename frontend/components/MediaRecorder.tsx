"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Mic, Video, Square, RotateCcw, AlertCircle, Check } from "lucide-react";

interface MediaRecorderProps {
  onRecordingComplete: (file: File, type: "audio" | "video") => void;
  disabled?: boolean;
}

type RecordingMode = "audio" | "video" | null;
type RecordingState = "idle" | "recording" | "done";

export default function MediaRecorderComponent({
  onRecordingComplete,
  disabled = false,
}: MediaRecorderProps) {
  const [mode, setMode] = useState<RecordingMode>(null);
  const [state, setState] = useState<RecordingState>("idle");
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const liveVideoRef = useRef<HTMLVideoElement>(null);

  // Cleanup function
  const cleanup = useCallback((urlToRevoke?: string) => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (urlToRevoke) {
      URL.revokeObjectURL(urlToRevoke);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanup();
    };
  }, []);

  // Connect live video stream when recording starts
  useEffect(() => {
    if (state === "recording" && mode === "video" && liveVideoRef.current && streamRef.current) {
      liveVideoRef.current.srcObject = streamRef.current;
      liveVideoRef.current.play().catch(console.error);
    }
  }, [state, mode]);

  const startRecording = async (recordMode: RecordingMode) => {
    if (!recordMode) return;

    setError(null);
    chunksRef.current = [];

    try {
      const constraints: MediaStreamConstraints =
        recordMode === "video"
          ? { video: { facingMode: "user", width: 1280, height: 720 }, audio: true }
          : { audio: true };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      // Set mode and state first so video element renders
      setMode(recordMode);
      setState("recording");
      setDuration(0);

      const mimeType = recordMode === "video"
        ? (MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
          ? "video/webm;codecs=vp9"
          : "video/webm")
        : (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : "audio/webm");

      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        setRecordedBlob(blob);
        setState("done");

        // Stop all tracks
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
        }
      };

      mediaRecorder.start(100); // Collect data every 100ms

      // Start timer
      timerRef.current = setInterval(() => {
        setDuration((prev) => prev + 1);
      }, 1000);
    } catch (err: any) {
      console.error("Error accessing media devices:", err);
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        setError("Permission denied. Allow mic/camera access and refresh.");
      } else if (err.name === "NotFoundError") {
        setError("No microphone or camera found.");
      } else {
        setError("Failed to access recording device.");
      }
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && state === "recording") {
      mediaRecorderRef.current.stop();
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  const resetRecording = () => {
    cleanup();
    setMode(null);
    setState("idle");
    setRecordedBlob(null);
    setDuration(0);
    setError(null);
  };

  const submitRecording = useCallback(() => {
    if (!recordedBlob || !mode) return;

    const filename = `recording-${Date.now()}.webm`;
    const file = new File([recordedBlob], filename, { type: recordedBlob.type });

    onRecordingComplete(file, mode);
    resetRecording();
  }, [recordedBlob, mode, onRecordingComplete]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  // Idle state - show record options
  if (state === "idle") {
    return (
      <div className="space-y-3">
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="text-sm">{error}</AlertDescription>
          </Alert>
        )}

        <div className="grid grid-cols-2 gap-3">
          <Button
            onClick={() => startRecording("audio")}
            disabled={disabled}
            variant="outline"
            className="h-20 flex-col gap-1.5"
          >
            <Mic className="w-6 h-6" />
            <span className="text-sm">Audio</span>
          </Button>
          <Button
            onClick={() => startRecording("video")}
            disabled={disabled}
            variant="outline"
            className="h-20 flex-col gap-1.5"
          >
            <Video className="w-6 h-6" />
            <span className="text-sm">Video</span>
          </Button>
        </div>
      </div>
    );
  }

  // Recording state
  if (state === "recording") {
    return (
      <div className="space-y-3">
        {/* Live video preview */}
        {mode === "video" && (
          <div className="rounded-lg overflow-hidden bg-muted border" style={{ maxHeight: "280px" }}>
            <video
              ref={liveVideoRef}
              autoPlay
              muted
              playsInline
              className="w-full h-full object-contain"
              style={{ transform: "scaleX(-1)", maxHeight: "280px" }}
            />
          </div>
        )}

        {/* Audio recording indicator */}
        {mode === "audio" && (
          <div className="flex items-center justify-center h-24 bg-muted/50 rounded-lg border">
            <div className="flex items-center gap-3">
              <div className="relative">
                <Mic className="w-8 h-8 text-red-500" />
                <span className="absolute -top-0.5 -right-0.5 flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span>
                </span>
              </div>
              <div className="text-left">
                <p className="text-sm font-medium">Recording...</p>
                <p className="text-xs text-muted-foreground">{formatTime(duration)}</p>
              </div>
            </div>
          </div>
        )}

        {/* Recording controls */}
        <div className="flex items-center justify-between">
          {mode === "video" && (
            <Badge variant="destructive" className="animate-pulse">
              <span className="mr-1.5">●</span>
              {formatTime(duration)}
            </Badge>
          )}
          {mode === "audio" && <div />}

          <Button
            onClick={stopRecording}
            variant="destructive"
            className="gap-2"
          >
            <Square className="w-4 h-4 fill-current" />
            Stop
          </Button>
        </div>
      </div>
    );
  }

  // Done state - recording finished, show confirmation
  if (state === "done" && recordedBlob) {
    return (
      <div className="space-y-3">
        {/* Recording complete indicator */}
        <div className="flex items-center justify-center h-20 bg-muted/50 rounded-lg border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
              <Check className="w-5 h-5 text-green-600 dark:text-green-400" />
            </div>
            <div className="text-left">
              <p className="text-sm font-medium">Recording complete</p>
              <p className="text-xs text-muted-foreground">
                {mode === "video" ? "Video" : "Audio"} • {formatTime(duration)}
              </p>
            </div>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex gap-2">
          <Button
            onClick={resetRecording}
            variant="outline"
            className="flex-1 gap-1.5"
          >
            <RotateCcw className="w-4 h-4" />
            Again
          </Button>
          <Button
            onClick={submitRecording}
            className="flex-1 gap-1.5"
          >
            <Check className="w-4 h-4" />
            Use This
          </Button>
        </div>
      </div>
    );
  }

  return null;
}

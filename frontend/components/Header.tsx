"use client";

import { Badge } from "@/components/ui/badge";
import { MonitorSpeaker, ArrowRight } from "lucide-react";

export default function Header() {
  return (
    <section className="bg-background border-b">
      <div className="max-w-7xl mx-auto px-3 sm:px-6 lg:px-8 py-4 sm:py-6">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <MonitorSpeaker className="w-5 h-5 sm:w-6 sm:h-6 text-primary" />
            <ArrowRight className="w-4 h-4 text-muted-foreground" />
            <p className="text-sm sm:text-base text-foreground">
              Upload audio or video in Sundanese, get transcription and Indonesian translation
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="text-xs">
              Whisper ASR
            </Badge>
            <Badge variant="outline" className="text-xs">
              NLLB Translation
            </Badge>
          </div>
        </div>
      </div>
    </section>
  );
}

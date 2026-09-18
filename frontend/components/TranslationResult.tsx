"use client";

import { Copy, Download, CheckCheck } from "lucide-react";
import { useState } from "react";
import { copyToClipboard, downloadText, formatDuration } from "@/lib/utils";
import type { TranslateResponse } from "@/types";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";

interface TranslationResultProps {
  result: TranslateResponse;
}

export default function TranslationResult({ result }: TranslationResultProps) {
  const [copiedOcr, setCopiedOcr] = useState(false);
  const [copiedTranslation, setCopiedTranslation] = useState(false);
  const { toast } = useToast();

  const handleCopy = async (text: string, type: "ocr" | "translation") => {
    const success = await copyToClipboard(text);
    if (success) {
      toast({
        title: "Copied!",
        description: "Text copied to clipboard",
      });
      if (type === "ocr") {
        setCopiedOcr(true);
        setTimeout(() => setCopiedOcr(false), 2000);
      } else {
        setCopiedTranslation(true);
        setTimeout(() => setCopiedTranslation(false), 2000);
      }
    }
  };

  const handleDownload = () => {
    const detectionInfo = result.detection
      ? `\nDetection Info:\n- Detected Script: ${result.detection.detected_script}\n- Detection Confidence: ${result.detection.confidence.toFixed(2)}%\n- Detection Time: ${formatDuration(result.detection.time)}\n`
      : "";
    const ocrMethodInfo = result.ocr.method
      ? `- OCR Method: ${result.ocr.method}\n`
      : "";

    const content = `OCR Text (Sundanese):\n${result.ocr.text}\n\nTranslation (Indonesian):\n${result.translation.translated}\n\nProcessing Info:${detectionInfo}${ocrMethodInfo}- OCR Time: ${formatDuration(result.ocr.time)}\n- Translation Time: ${formatDuration(result.translation.time)}\n- Total Time: ${formatDuration(result.total_time)}\n- OCR Confidence: ${result.ocr.confidence.toFixed(2)}%`;

    downloadText(content, `translation_${Date.now()}.txt`);
    toast({
      title: "Downloaded!",
      description: "Translation result has been downloaded",
    });
  };

  const getScriptIcon = (script: string) => {
    switch (script) {
      case "latin":
        return "📝";
      case "aksara_sunda":
        return "🔤";
      default:
        return "📄";
    }
  };

  const getMethodName = (method: string) => {
    switch (method) {
      case "tesseract":
        return "Tesseract OCR (Latin Script)";
      case "aksara_sunda_handwritten":
        return "Aksara Sunda CNN (Handwritten)";
      case "aksara_sunda_unicode":
        return "Aksara Sunda CNN (Unicode/Digital)";
      default:
        return method;
    }
  };

  return (
    <div className="space-y-6">
      {/* Detection Info */}
      {result.detection && (
        <Card className="bg-gradient-to-br from-blue-50 to-card shadow-none border-blue-200">
          <CardHeader className="pb-3 pt-4">
            <CardTitle className="text-sm flex items-center gap-2">
              <span className="text-2xl">
                {getScriptIcon(result.detection.detected_script)}
              </span>
              Auto-Detection Result
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Detected Script:</span>
                <Badge variant="secondary" className="font-medium">
                  {result.detection.detected_script === "latin"
                    ? "Latin Script"
                    : "Aksara Sunda"}
                </Badge>
              </div>
              {result.ocr.method && (
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">OCR Method:</span>
                  <span className="font-medium text-xs">
                    {getMethodName(result.ocr.method)}
                  </span>
                </div>
              )}
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">
                  Detection Confidence:
                </span>
                <span className="font-medium">
                  {result.detection.confidence.toFixed(1)}%
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Processing Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-gradient-to-br from-primary/5 to-card shadow-none border-primary/10">
          <CardHeader className="pb-3 pt-4">
            <CardDescription className="text-xs">OCR Time</CardDescription>
            <CardTitle className="text-2xl font-bold tabular-nums mt-2">
              {formatDuration(result.ocr.time)}
            </CardTitle>
          </CardHeader>
        </Card>

        <Card className="bg-gradient-to-br from-primary/5 to-card shadow-none border-primary/10">
          <CardHeader className="pb-3 pt-4">
            <CardDescription className="text-xs">
              Translation Time
            </CardDescription>
            <CardTitle className="text-2xl font-bold tabular-nums mt-2">
              {formatDuration(result.translation.time)}
            </CardTitle>
          </CardHeader>
        </Card>

        <Card className="bg-gradient-to-br from-primary/5 to-card shadow-none border-primary/10">
          <CardHeader className="pb-3 pt-4">
            <CardDescription className="text-xs">Total Time</CardDescription>
            <CardTitle className="text-2xl font-bold tabular-nums mt-2">
              {formatDuration(result.total_time)}
            </CardTitle>
          </CardHeader>
        </Card>

        <Card className="bg-gradient-to-br from-primary/5 to-card shadow-none border-primary/10">
          <CardHeader className="pb-3 pt-4">
            <CardDescription className="text-xs">Confidence</CardDescription>
            <CardTitle className="text-2xl font-bold tabular-nums mt-2">
              {result.ocr.confidence.toFixed(1)}%
            </CardTitle>
          </CardHeader>
        </Card>
      </div>

      {/* OCR Result */}
      <Card className="shadow-none">
        <CardHeader className="pb-3 pt-4">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Extracted Text</CardTitle>
              <CardDescription>Sundanese text from image</CardDescription>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => handleCopy(result.ocr.text, "ocr")}
              title="Copy to clipboard"
            >
              {copiedOcr ? (
                <CheckCheck className="w-4 h-4 text-green-600" />
              ) : (
                <Copy className="w-4 h-4" />
              )}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="bg-muted rounded-lg p-4">
            <p className="text-foreground whitespace-pre-wrap">
              {result.ocr.text}
            </p>
          </div>
          <div className="mt-3 flex gap-2">
            <Badge variant="outline">
              {result.text_stats.word_count} words
            </Badge>
            <Badge variant="outline">
              {result.text_stats.length} characters
            </Badge>
          </div>
        </CardContent>
      </Card>

      {/* Translation Result */}
      <Card className="border-primary/30 shadow-none">
        <CardHeader className="pb-3 pt-4">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-primary">Translation</CardTitle>
              <CardDescription>Indonesian translation</CardDescription>
            </div>
            <div className="flex gap-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={() =>
                  handleCopy(result.translation.translated, "translation")
                }
                title="Copy to clipboard"
              >
                {copiedTranslation ? (
                  <CheckCheck className="w-4 h-4 text-green-600" />
                ) : (
                  <Copy className="w-4 h-4" />
                )}
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={handleDownload}
                title="Download result"
              >
                <Download className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="bg-primary/10 rounded-lg p-4 border border-primary/20">
            <p className="text-foreground whitespace-pre-wrap font-medium">
              {result.translation.translated}
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

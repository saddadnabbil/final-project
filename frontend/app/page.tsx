"use client";

import { useState, useEffect } from "react";
import AudioUpload from "@/components/AudioUpload";
import AudioVideoResult from "@/components/AudioVideoResult";
import MediaRecorderComponent from "@/components/MediaRecorder";
import ResultSkeleton from "@/components/ResultSkeleton";
import ErrorMessage from "@/components/ErrorMessage";
import TranslationHistory from "@/components/TranslationHistory";
import { useAuth } from "@/contexts/AuthContext";
import { apiClient } from "@/lib/api";
import type { AudioVideoResponse } from "@/types";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Upload,
  FileAudio,
  Video,
  Languages,
  History,
  Info,
  Sparkles,
  Heart,
  X,
  Mic,
} from "lucide-react";

export default function Home() {
  const [activeTab, setActiveTab] = useState<string>("translate");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileType, setFileType] = useState<"audio" | "video" | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<AudioVideoResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [processingStage, setProcessingStage] = useState<string>("");
  const [history, setHistory] = useState<any[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyPage, setHistoryPage] = useState(1);
  const [historyPagination, setHistoryPagination] = useState<any>(null);

  const { isAuthenticated, refreshUser } = useAuth();

  const handleFileSelect = (file: File) => {
    // Clean up previous preview URL
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    // Determine file type
    const isVideo = file.type.startsWith("video/");
    const isAudio = file.type.startsWith("audio/");

    if (!isVideo && !isAudio) {
      setError("Please select an audio or video file");
      return;
    }

    // Reset states
    setSelectedFile(file);
    setFileType(isVideo ? "video" : "audio");
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleRemoveFile = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(null);
    setFileType(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const handleProcess = async () => {
    if (!selectedFile || !fileType) return;

    setIsProcessing(true);
    setError(null);
    setResult(null);

    try {
      if (fileType === "audio") {
        setProcessingStage("Uploading audio file...");
        await new Promise((resolve) => setTimeout(resolve, 100));

        setProcessingStage("Transcribing audio...");
        await new Promise((resolve) => setTimeout(resolve, 100));

        setProcessingStage("Translating...");
        const response = await apiClient.uploadAudio(selectedFile);
        setResult(response);
      } else {
        setProcessingStage("Uploading video file...");
        await new Promise((resolve) => setTimeout(resolve, 100));

        setProcessingStage("Extracting audio from video...");
        await new Promise((resolve) => setTimeout(resolve, 100));

        setProcessingStage("Processing audio-visual features...");
        await new Promise((resolve) => setTimeout(resolve, 100));

        setProcessingStage("Transcribing and translating...");
        const response = await apiClient.uploadVideo(selectedFile);
        setResult(response);
      }

      // Refresh user data to update translation count in header
      if (isAuthenticated) {
        await refreshUser();
      }
    } catch (err: any) {
      console.error("Processing error:", err);
      const errorMessage =
        err.response?.data?.error ||
        err.message ||
        "An error occurred during processing";
      setError(errorMessage);
    } finally {
      setIsProcessing(false);
      setProcessingStage("");
    }
  };

  const handleRetry = () => {
    setError(null);
    handleProcess();
  };

  const fetchHistory = async (page: number = 1) => {
    if (!isAuthenticated) return;
    
    setHistoryLoading(true);
    try {
      const response = await apiClient.getTranslationHistory(page, 10);
      setHistory(response.histories || []);
      setHistoryPagination(response.pagination || null);
      setHistoryPage(page);
    } catch (error) {
      console.error('Failed to fetch history:', error);
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  const deleteHistoryItem = async (historyId: number) => {
    try {
      await apiClient.deleteTranslationHistory(historyId);
      // Refresh history after deletion
      fetchHistory(historyPage);
    } catch (error) {
      console.error('Failed to delete history item:', error);
    }
  };

  // Fetch history when user logs in or tab changes to history
  useEffect(() => {
    if (isAuthenticated && activeTab === 'history') {
      fetchHistory(1);
    }
  }, [isAuthenticated, activeTab]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-background to-muted/20">
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className={`grid w-full max-w-lg mx-auto mb-8 ${isAuthenticated ? 'grid-cols-3' : 'grid-cols-2'}`}>
            <TabsTrigger value="translate" className="flex items-center gap-2">
              <Upload className="w-4 h-4" />
              Upload
            </TabsTrigger>
            <TabsTrigger value="record" className="flex items-center gap-2">
              <Mic className="w-4 h-4" />
              Record
            </TabsTrigger>
            {isAuthenticated && (
              <TabsTrigger value="history" className="flex items-center gap-2">
                <History className="w-4 h-4" />
                History
              </TabsTrigger>
            )}
          </TabsList>

          {/* Translate Tab */}
          <TabsContent value="translate" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left Column - Upload & Preview */}
              <div className="space-y-6">
                <Card className="shadow-none">
                  <CardHeader className="pb-3 pt-4">
                    <CardTitle className="flex items-center gap-2">
                      <Upload className="w-5 h-5" />
                      Upload Audio/Video
                    </CardTitle>
                    <CardDescription>
                      Upload audio or video file containing Sundanese speech
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {!selectedFile ? (
                      <div className="space-y-4">
                        {/* Audio-Visual Info */}
                        <Alert className="border-primary/20 bg-primary/5">
                          <Sparkles className="h-4 w-4" />
                          <AlertDescription className="text-sm">
                            <strong>Audio-Visual Processing:</strong> Upload audio
                            for transcription or video for enhanced audio-visual
                            translation of Sundanese speech.
                          </AlertDescription>
                        </Alert>

                        <AudioUpload
                          onFileSelect={handleFileSelect}
                          disabled={isProcessing}
                          acceptVideo={true}
                        />
                      </div>
                    ) : (
                      <div className="space-y-4">
                        {/* Preview */}
                        {previewUrl && fileType === "video" && (
                          <div className="rounded-lg overflow-hidden bg-black">
                            <video
                              src={previewUrl}
                              controls
                              className="w-full aspect-video"
                            />
                          </div>
                        )}

                        {previewUrl && fileType === "audio" && (
                          <div className="p-6 border rounded-lg bg-muted/30">
                            <div className="flex items-center gap-3 mb-4">
                              <FileAudio className="w-8 h-8 text-primary" />
                              <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium truncate">
                                  {selectedFile?.name}
                                </p>
                                <p className="text-xs text-muted-foreground">
                                  {(
                                    (selectedFile?.size || 0) /
                                    1024 /
                                    1024
                                  ).toFixed(2)}{" "}
                                  MB
                                </p>
                              </div>
                            </div>
                            <audio src={previewUrl} controls className="w-full" />
                          </div>
                        )}

                        <Button
                          onClick={handleProcess}
                          disabled={isProcessing}
                          className="w-full"
                          size="lg"
                        >
                          <Languages className="w-4 h-4 mr-2" />
                          {isProcessing ? "Processing..." : "Process & Translate"}
                        </Button>

                        {!isProcessing && (
                          <Button
                            onClick={handleRemoveFile}
                            variant="outline"
                            className="w-full"
                          >
                            <X className="w-4 h-4 mr-2" />
                            Choose Another File
                          </Button>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Instructions */}
                <Card className="border-primary/20 bg-gradient-to-br from-primary/5 to-card shadow-none">
                  <CardHeader className="pb-3 pt-4">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Info className="w-5 h-5" />
                      How to use
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ol className="space-y-2 text-sm text-muted-foreground">
                      <li className="flex gap-2">
                        <span className="font-semibold text-primary">1.</span>
                        <span>
                          Upload an audio or video file with Sundanese speech
                        </span>
                      </li>
                      <li className="flex gap-2">
                        <span className="font-semibold text-primary">2.</span>
                        <span>
                          System processes audio and visual features (if video)
                        </span>
                      </li>
                      <li className="flex gap-2">
                        <span className="font-semibold text-primary">3.</span>
                        <span>
                          Click "Process" to transcribe and translate
                        </span>
                      </li>
                      <li className="flex gap-2">
                        <span className="font-semibold text-primary">4.</span>
                        <span>View transcription and translation results</span>
                      </li>
                    </ol>
                  </CardContent>
                </Card>
              </div>

              {/* Right Column - Results */}
              <div className="space-y-6">
                <Card className="min-h-[500px] shadow-none">
                  <CardHeader className="pb-3 pt-4">
                    <CardTitle>Results</CardTitle>
                    <CardDescription>
                      Transcription and translation will appear here
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {isProcessing && (
                      <ResultSkeleton message={processingStage} />
                    )}

                    {error && (
                      <ErrorMessage message={error} onRetry={handleRetry} />
                    )}

                    {result && !isProcessing && !error && (
                      <AudioVideoResult result={result} />
                    )}

                    {!isProcessing && !error && !result && (
                      <div className="flex items-center justify-center h-[400px] text-muted-foreground">
                        <div className="text-center space-y-3">
                          <div className="flex justify-center gap-4">
                            <FileAudio className="w-12 h-12 opacity-50" />
                            <Video className="w-12 h-12 opacity-50" />
                          </div>
                          <div>
                            <p className="text-lg font-medium">
                              No file uploaded yet
                            </p>
                            <p className="text-sm">
                              Upload audio or video to get started
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Record Tab */}
          <TabsContent value="record" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left Column - Record */}
              <div className="space-y-6">
                <Card className="shadow-none">
                  <CardHeader className="pb-3 pt-4">
                    <CardTitle className="flex items-center gap-2">
                      <Mic className="w-5 h-5" />
                      Record Audio/Video
                    </CardTitle>
                    <CardDescription>
                      Record audio or video containing Sundanese speech
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {!selectedFile ? (
                      <div className="space-y-4">
                        {/* Recording Info */}
                        <Alert className="border-primary/20 bg-primary/5">
                          <Sparkles className="h-4 w-4" />
                          <AlertDescription className="text-sm">
                            <strong>Direct Recording:</strong> Record audio or
                            video directly from your browser for Sundanese
                            speech translation.
                          </AlertDescription>
                        </Alert>

                        <MediaRecorderComponent
                          onRecordingComplete={(file, type) => {
                            setSelectedFile(file);
                            setFileType(type);
                            setPreviewUrl(URL.createObjectURL(file));
                            setResult(null);
                            setError(null);
                          }}
                          disabled={isProcessing}
                        />
                      </div>
                    ) : (
                      <div className="space-y-4">
                        {/* Preview */}
                        {previewUrl && fileType === "video" && (
                          <div className="rounded-lg overflow-hidden bg-black">
                            <video
                              src={previewUrl}
                              controls
                              className="w-full aspect-video"
                            />
                          </div>
                        )}

                        {previewUrl && fileType === "audio" && (
                          <div className="p-6 border rounded-lg bg-muted/30">
                            <div className="flex items-center gap-3 mb-4">
                              <FileAudio className="w-8 h-8 text-primary" />
                              <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium truncate">
                                  {selectedFile?.name}
                                </p>
                                <p className="text-xs text-muted-foreground">
                                  {(
                                    (selectedFile?.size || 0) /
                                    1024 /
                                    1024
                                  ).toFixed(2)}{" "}
                                  MB
                                </p>
                              </div>
                            </div>
                            <audio src={previewUrl} controls className="w-full" />
                          </div>
                        )}

                        <Button
                          onClick={handleProcess}
                          disabled={isProcessing}
                          className="w-full"
                          size="lg"
                        >
                          <Languages className="w-4 h-4 mr-2" />
                          {isProcessing ? "Processing..." : "Process & Translate"}
                        </Button>

                        {!isProcessing && (
                          <Button
                            onClick={handleRemoveFile}
                            variant="outline"
                            className="w-full"
                          >
                            <X className="w-4 h-4 mr-2" />
                            Record Again
                          </Button>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Instructions */}
                <Card className="border-primary/20 bg-gradient-to-br from-primary/5 to-card shadow-none">
                  <CardHeader className="pb-3 pt-4">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Info className="w-5 h-5" />
                      How to use
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ol className="space-y-2 text-sm text-muted-foreground">
                      <li className="flex gap-2">
                        <span className="font-semibold text-primary">1.</span>
                        <span>
                          Choose to record audio or video
                        </span>
                      </li>
                      <li className="flex gap-2">
                        <span className="font-semibold text-primary">2.</span>
                        <span>
                          Allow microphone/camera access when prompted
                        </span>
                      </li>
                      <li className="flex gap-2">
                        <span className="font-semibold text-primary">3.</span>
                        <span>
                          Speak in Sundanese, then click "Stop"
                        </span>
                      </li>
                      <li className="flex gap-2">
                        <span className="font-semibold text-primary">4.</span>
                        <span>Preview and click "Process & Translate"</span>
                      </li>
                    </ol>
                  </CardContent>
                </Card>
              </div>

              {/* Right Column - Results */}
              <div className="space-y-6">
                <Card className="min-h-[500px] shadow-none">
                  <CardHeader className="pb-3 pt-4">
                    <CardTitle>Results</CardTitle>
                    <CardDescription>
                      Transcription and translation will appear here
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {isProcessing && (
                      <ResultSkeleton message={processingStage} />
                    )}

                    {error && (
                      <ErrorMessage message={error} onRetry={handleRetry} />
                    )}

                    {result && !isProcessing && !error && (
                      <AudioVideoResult result={result} />
                    )}

                    {!isProcessing && !error && !result && (
                      <div className="flex items-center justify-center h-[400px] text-muted-foreground">
                        <div className="text-center space-y-3">
                          <div className="flex justify-center gap-4">
                            <Mic className="w-12 h-12 opacity-50" />
                            <Video className="w-12 h-12 opacity-50" />
                          </div>
                          <div>
                            <p className="text-lg font-medium">
                              No recording yet
                            </p>
                            <p className="text-sm">
                              Record audio or video to get started
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* History Tab */}
          <TabsContent value="history">
            {isAuthenticated ? (
              <TranslationHistory 
                history={history}
                loading={historyLoading}
                onDeleteItem={deleteHistoryItem}
                pagination={historyPagination}
                onPageChange={fetchHistory}
              />
            ) : (
              <Card className="shadow-none">
                <CardContent className="py-12 px-6">
                  <div className="text-center space-y-3">
                    <History className="w-12 h-12 mx-auto text-muted-foreground" />
                    <div>
                      <p className="text-lg font-medium">Login Required</p>
                      <p className="text-sm text-muted-foreground mt-2">
                        Please login to view your translation history
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>

        {/* Footer Info */}
        <div className="mt-12 pt-8 border-t">
          <div className="text-center text-sm text-muted-foreground space-y-2">
            <p className="flex items-center justify-center gap-1.5">
              Made with <Heart className="w-4 h-4 text-red-500 fill-red-500" /> by 
            </p>
            {isAuthenticated && (
              <p className="text-xs">
                Your translation history is saved automatically when you're
                logged in
              </p>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  RefreshCw,
  Trash2,
  AlertCircle,
  Clock,
  FileText,
  Calendar,
  TrendingUp,
  Timer,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface HistoryMetrics {
  processing_time?: number;
  audio_duration?: number;
}

interface HistoryItem {
  id: number;
  content_type: string;
  original_filename: string;
  file_size: number;
  original_text: string;
  translated_text: string;
  processing_time: number;
  audio_duration?: number;
  mode: string;
  metrics?: HistoryMetrics;
  created_at: string;
}

interface HistoryStats {
  total_translations: number;
  avg_processing_time: number;
  total_characters_translated: number;
  recent_translations_7days: number;
}

export default function TranslationHistory() {
  const { token, refreshUser } = useAuth();
  const { toast } = useToast();
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [stats, setStats] = useState<HistoryStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedItem, setSelectedItem] = useState<HistoryItem | null>(null);
  const [hasLoadedOnce, setHasLoadedOnce] = useState(false);
  const [filter, setFilter] = useState<string>("all");

  // Normalize API URL by removing trailing slashes
  const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001").replace(/\/+$/, '');

  useEffect(() => {
    if (token && !hasLoadedOnce) {
      fetchHistory();
      fetchStats();
    } else if (token && hasLoadedOnce) {
      setIsLoading(false);
    }
  }, [token, hasLoadedOnce]);

  const fetchHistory = async () => {
    try {
      const response = await fetch(`${API_URL}/api/translation-history`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      const data = await response.json();

      if (response.status === 401) {
        setError("Session expired. Please login again.");
        setTimeout(() => {
          localStorage.clear();
          window.location.reload();
        }, 2000);
        return;
      }

      if (data.success) {
        setHistory(data.histories);
        setHasLoadedOnce(true);
      } else {
        setError(data.error || "Failed to load history");
      }
    } catch (err: any) {
      console.error("Fetch history error:", err);
      setError("Failed to fetch history");
    } finally {
      setIsLoading(false);
    }
  };

  const refreshHistory = () => {
    setIsLoading(true);
    setError("");
    fetchHistory();
    fetchStats();
  };

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_URL}/api/profile`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.status === 401) {
        return;
      }

      const data = await response.json();

      if (data.user && data.user.stats) {
        setStats(data.user.stats);
      }
    } catch (err) {
      console.error("Failed to fetch stats:", err);
    }
  };

  const deleteItem = async (id: number) => {
    try {
      const response = await fetch(`${API_URL}/api/translation-history/${id}`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      const data = await response.json();

      if (data.message) {
        setHistory(history.filter((item) => item.id !== id));
        if (selectedItem?.id === id) {
          setSelectedItem(null);
        }
        fetchStats();
        await refreshUser();
        toast({
          title: "Deleted",
          description: "History item has been deleted",
        });
      }
    } catch (err) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to delete item",
      });
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString("id-ID", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getTypeLabel = (type: string) => {
    switch (type) {
      case "audio":
        return "Audio Translation";
      case "video":
        return "Video Translation";
      default:
        return type;
    }
  };

  const filteredHistory =
    filter === "all"
      ? history
      : history.filter((item) => item.content_type === filter);

  if (isLoading) {
    return (
      <div className="w-full max-w-6xl mx-auto p-6">
        <div className="mb-6">
          <Skeleton className="h-8 w-48 mb-4" />

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            {[1, 2, 3, 4].map((i) => (
              <Card
                key={i}
                className="bg-gradient-to-br from-primary/5 to-card shadow-none border-primary/10"
              >
                <CardHeader className="pb-3 pt-4">
                  <Skeleton className="h-4 w-32 mb-2" />
                  <Skeleton className="h-8 w-16" />
                </CardHeader>
              </Card>
            ))}
          </div>

          <Skeleton className="h-10 w-32" />
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} className="shadow-none">
              <CardHeader className="pb-3 pt-4">
                <div className="flex justify-between items-start mb-2">
                  <div className="flex-1">
                    <Skeleton className="h-4 w-32 mb-2" />
                    <Skeleton className="h-3 w-48" />
                  </div>
                  <Skeleton className="h-4 w-12" />
                </div>
                <div className="space-y-2 mt-4">
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-3/4" />
                </div>
                <div className="flex justify-between items-center mt-3 pt-3 border-t">
                  <Skeleton className="h-3 w-24" />
                  <Skeleton className="h-3 w-16" />
                </div>
              </CardHeader>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6">
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold">Translation History</h2>
          <Button onClick={refreshHistory} variant="outline" size="sm">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>

        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <Card className="bg-gradient-to-br from-primary/5 to-card shadow-none border-primary/10">
              <CardHeader className="pb-3 pt-4">
                <CardDescription className="flex items-center gap-2 text-xs">
                  <TrendingUp className="w-4 h-4" />
                  Total Translations
                </CardDescription>
                <CardTitle className="text-3xl font-bold tabular-nums mt-2">
                  {stats.total_translations}
                </CardTitle>
              </CardHeader>
            </Card>

            <Card className="bg-gradient-to-br from-primary/5 to-card shadow-none border-primary/10">
              <CardHeader className="pb-3 pt-4">
                <CardDescription className="flex items-center gap-2 text-xs">
                  <Clock className="w-4 h-4" />
                  Avg Time
                </CardDescription>
                <CardTitle className="text-3xl font-bold tabular-nums mt-2">
                  {stats.avg_processing_time.toFixed(2)}s
                </CardTitle>
              </CardHeader>
            </Card>

            <Card className="bg-gradient-to-br from-primary/5 to-card shadow-none border-primary/10">
              <CardHeader className="pb-3 pt-4">
                <CardDescription className="flex items-center gap-2 text-xs">
                  <FileText className="w-4 h-4" />
                  Total Characters
                </CardDescription>
                <CardTitle className="text-3xl font-bold tabular-nums mt-2">
                  {stats.total_characters_translated}
                </CardTitle>
              </CardHeader>
            </Card>

            <Card className="bg-gradient-to-br from-primary/5 to-card shadow-none border-primary/10">
              <CardHeader className="pb-3 pt-4">
                <CardDescription className="flex items-center gap-2 text-xs">
                  <Calendar className="w-4 h-4" />
                  Last 7 Days
                </CardDescription>
                <CardTitle className="text-3xl font-bold tabular-nums mt-2">
                  {stats.recent_translations_7days}
                </CardTitle>
              </CardHeader>
            </Card>
          </div>
        )}

        <div className="flex gap-2 mb-4">
          <Button
            variant={filter === "all" ? "default" : "outline"}
            size="sm"
            onClick={() => setFilter("all")}
          >
            All
          </Button>
          <Button
            variant={filter === "audio" ? "default" : "outline"}
            size="sm"
            onClick={() => setFilter("audio")}
          >
            Audio
          </Button>
          <Button
            variant={filter === "video" ? "default" : "outline"}
            size="sm"
            onClick={() => setFilter("video")}
          >
            Video
          </Button>
        </div>
      </div>

      {error && (
        <Alert variant="destructive" className="mb-4">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {filteredHistory.length === 0 ? (
        <Card className="shadow-none">
          <CardContent className="py-12 px-6 text-center">
            <p className="text-muted-foreground">
              No translation history yet. Start translating to see your history
              here!
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid md:grid-cols-2 gap-4">
          {filteredHistory.map((item) => (
            <Card
              key={item.id}
              className="cursor-pointer hover:border-primary/30 transition-colors shadow-none"
              onClick={() => setSelectedItem(item)}
            >
              <CardHeader className="pb-3 pt-4">
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <Badge variant="outline">
                        {getTypeLabel(item.content_type)}
                      </Badge>
                      <Badge variant="secondary">{item.mode}</Badge>
                    </div>
                    <CardDescription className="text-xs">
                      {formatDate(item.created_at)}
                    </CardDescription>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (
                        confirm("Are you sure you want to delete this item?")
                      ) {
                        deleteItem(item.id);
                      }
                    }}
                  >
                    <Trash2 className="w-4 h-4 text-destructive" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="space-y-3">
                  <div>
                    <p className="text-xs font-medium text-muted-foreground">
                      Original Text:
                    </p>
                    <p className="text-sm line-clamp-2">{item.original_text}</p>
                  </div>
                  <div>
                    <p className="text-xs font-medium text-muted-foreground">
                      Translation:
                    </p>
                    <p className="text-sm line-clamp-2">
                      {item.translated_text}
                    </p>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2 mt-3 pt-3 border-t">
                  {item.processing_time && (
                    <Badge variant="outline" className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {item.processing_time.toFixed(2)}s
                    </Badge>
                  )}
                  {item.audio_duration && (
                    <Badge variant="outline" className="flex items-center gap-1">
                      <Timer className="w-3 h-3" />
                      {item.audio_duration.toFixed(1)}s audio
                    </Badge>
                  )}
                  {item.file_size && (
                    <Badge variant="outline">
                      {(item.file_size / 1024).toFixed(1)} KB
                    </Badge>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      <Dialog open={!!selectedItem} onOpenChange={() => setSelectedItem(null)}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Translation Details</DialogTitle>
            <DialogDescription>
              View complete translation information
            </DialogDescription>
          </DialogHeader>

          {selectedItem && (
            <div className="space-y-4">
              <div>
                <p className="text-sm font-medium text-muted-foreground mb-1">
                  Date:
                </p>
                <p>{formatDate(selectedItem.created_at)}</p>
              </div>

              <div className="flex gap-2">
                <Badge>{getTypeLabel(selectedItem.content_type)}</Badge>
                <Badge variant="secondary">{selectedItem.mode}</Badge>
              </div>

              {selectedItem.original_filename && (
                <div>
                  <p className="text-sm font-medium text-muted-foreground mb-1">
                    File:
                  </p>
                  <p className="text-sm">{selectedItem.original_filename}</p>
                </div>
              )}

              <div>
                <p className="text-sm font-medium text-muted-foreground mb-1">
                  Original Text (Sundanese):
                </p>
                <Card>
                  <CardContent className="p-3 bg-muted">
                    <p className="text-sm whitespace-pre-wrap">
                      {selectedItem.original_text}
                    </p>
                  </CardContent>
                </Card>
              </div>

              <div>
                <p className="text-sm font-medium text-muted-foreground mb-1">
                  Translation (Indonesian):
                </p>
                <Card className="border-primary/50">
                  <CardContent className="p-3 bg-primary/10">
                    <p className="text-sm whitespace-pre-wrap">
                      {selectedItem.translated_text}
                    </p>
                  </CardContent>
                </Card>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {selectedItem.processing_time && (
                  <div>
                    <p className="text-sm font-medium text-muted-foreground mb-1">
                      Processing Time:
                    </p>
                    <Badge variant="secondary" className="flex items-center gap-1 w-fit">
                      <Clock className="w-3 h-3" />
                      {selectedItem.processing_time.toFixed(2)}s
                    </Badge>
                  </div>
                )}
                {selectedItem.audio_duration && (
                  <div>
                    <p className="text-sm font-medium text-muted-foreground mb-1">
                      Audio Duration:
                    </p>
                    <Badge variant="secondary" className="flex items-center gap-1 w-fit">
                      <Timer className="w-3 h-3" />
                      {selectedItem.audio_duration.toFixed(1)}s
                    </Badge>
                  </div>
                )}
                {selectedItem.file_size && (
                  <div>
                    <p className="text-sm font-medium text-muted-foreground mb-1">
                      File Size:
                    </p>
                    <Badge variant="secondary">
                      {(selectedItem.file_size / 1024).toFixed(1)} KB
                    </Badge>
                  </div>
                )}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
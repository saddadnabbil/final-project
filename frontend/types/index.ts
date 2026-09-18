// Audio/Video processing types

// Visual Augmentation data from backend (for BAB 4 documentation)
export interface VisualAugmentationData {
  enabled: boolean;
  lip_reading: {
    model: string;
    input_shape: number[];
    input_size: string;
    output_dim?: number;
  };
  facial_landmarks: {
    model: string;
    detected: boolean;
    landmark_count: number;
    feature_count: number;
  };
  fusion: {
    model: string;
    audio_dim: number;
    visual_dim: number;
    fusion_dim: number;
    output_dim?: number;
  };
  attention_weights: {
    audio: number;
    visual: number;
  };
  landmark_image?: string; // Base64 encoded image with landmark visualization
}

// Detailed WER metrics from backend
export interface WERMetrics {
  wer: number;
  accuracy: number;
  substitutions: number;
  deletions: number;
  insertions: number;
  total_words: number;
}

// Detailed CER metrics from backend
export interface CERMetrics {
  cer: number;
  accuracy: number;
  substitutions: number;
  deletions: number;
  insertions: number;
  total_chars: number;
}

// Detailed BLEU metrics from backend
export interface BLEUMetrics {
  bleu: number;
  brevity_penalty: number;
  precisions: number[];
}

export interface AudioVideoMetrics {
  processing_time?: number; // Processing time in seconds
  audio_duration?: number; // Audio duration in seconds
  wer?: WERMetrics; // Word Error Rate with details
  cer?: CERMetrics; // Character Error Rate with details
  bleu?: BLEUMetrics; // BLEU Score with details
}

export interface AudioVideoResponse {
  success: boolean;
  transcription: string;
  translation: string;
  mode: 'audio-only' | 'audio-visual';
  metrics?: AudioVideoMetrics;
  visual_augmentation?: VisualAugmentationData;
  error?: string;
}

// Legacy image types (kept for compatibility)
export interface ImageInfo {
  width: number;
  height: number;
  mode: string;
  format: string;
  file_size: number;
  file_size_mb: number;
}

export interface OCRResult {
  text: string;
  confidence: number;
  time: number;
  method?: string;
}

export interface TranslationResult {
  original: string;
  translated: string;
  time: number;
}

export interface TextStats {
  length: number;
  word_count: number;
  line_count: number;
  sentence_count: number;
  avg_word_length?: number;
}

export interface DetectionResult {
  detected_script: string;
  confidence: number;
  time: number;
}

export interface TranslateResponse {
  success: boolean;
  ocr: OCRResult;
  translation: TranslationResult;
  image_info: ImageInfo;
  total_time: number;
  text_stats: TextStats;
  detection?: DetectionResult;
  error?: string;
}

export interface HealthResponse {
  status: string;
  message: string;
  device: string;
  models_loaded: {
    ocr: boolean;
    translator: boolean;
  };
}

export interface ErrorResponse {
  error: string;
  details?: string;
}

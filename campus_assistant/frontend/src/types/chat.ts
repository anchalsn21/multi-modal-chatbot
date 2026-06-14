/**
 * TypeScript type definitions for the chat feature.
 */

/** A single message in the chat conversation */
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  modality?: "text" | "voice" | "image" | "multimodal";
  intent?: string;
  confidence?: number;
  imageMatch?: string;
  imageConfidence?: number;
  audioUrl?: string;
  imageUrl?: string;
  caption?: string;
  streaming?: boolean;
}

/** Response from POST /chat */
export interface ChatResponse {
  reply: string;
  intent: string;
  confidence: number;
  entity?: string;
}

/** Response from POST /chat/voice */
export interface VoiceResponse {
  reply: string;
  transcript: string;
}

/** Response from POST /chat/image */
export interface ImageResponse {
  reply: string;
  description: string;
  confidence: number;
}

/** Response from POST /chat/multimodal */
export interface MultimodalResponse {
  reply: string;
  intent: string;
  confidence: number;
  image_match: string | null;
  image_confidence: number | null;
}

/** Error shape returned by the API */
export interface ApiError {
  detail: string;
}

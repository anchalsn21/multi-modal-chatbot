/**
 * api.ts — Typed fetch wrappers for all backend API endpoints.
 *
 * All functions read NEXT_PUBLIC_API_URL from the environment so the backend
 * URL is never hardcoded. Set it in .env.local.
 */

import type { ChatResponse, VoiceResponse, ImageResponse, MultimodalResponse } from "@/types/chat";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** User-friendly translations for common HTTP error codes */
const ERROR_MESSAGES: Record<number, string> = {
  400: "Invalid request — please check your input.",
  415: "Please upload a JPEG or PNG image.",
  422: "The request could not be processed. Please try again.",
  500: "The assistant encountered an error. Please try again.",
  503: "The assistant is still starting up. Please wait a moment.",
};

async function apiFetch<T>(path: string, init: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new Error("Could not reach the campus assistant. Please check your connection.");
  }

  if (!response.ok) {
    const friendly = ERROR_MESSAGES[response.status];
    if (friendly) throw new Error(friendly);

    let detail = `HTTP ${response.status}`;
    try {
      const err = await response.json();
      detail = err.detail ?? detail;
    } catch {
      // ignore JSON parse error
    }
    throw new Error(detail);
  }

  return response.json() as Promise<T>;
}

/**
 * Send a typed text message to the campus assistant.
 */
export async function sendTextMessage(
  message: string,
  sessionId?: string,
): Promise<ChatResponse> {
  return apiFetch<ChatResponse>("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId ?? null }),
  });
}

/**
 * Stream a text reply word-by-word from the campus assistant.
 *
 * Calls onToken for each word as it arrives.
 * Calls onDone with intent/confidence/entity when the stream ends.
 */
export async function streamTextMessage(
  message: string,
  sessionId: string | undefined,
  onToken: (token: string) => void,
  onDone: (meta: { intent: string; confidence: number; entity: string }) => void,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId ?? null }),
    });
  } catch {
    throw new Error("Could not reach the campus assistant. Please check your connection.");
  }

  if (!response.ok) {
    const friendly = ERROR_MESSAGES[response.status];
    throw new Error(friendly ?? `HTTP ${response.status}`);
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE lines are separated by \n\n; process all complete events
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? ""; // last part may be incomplete

    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data:")) continue;
      const raw = line.slice(5).trim();
      try {
        const payload = JSON.parse(raw);
        if (payload.done) {
          onDone({ intent: payload.intent, confidence: payload.confidence, entity: payload.entity });
        } else if (payload.token !== undefined) {
          onToken(payload.token);
        }
      } catch {
        // malformed SSE line — skip
      }
    }
  }
}

/**
 * Transcribe a voice recording and return only the transcript string.
 * The NLP pipeline is NOT run — the caller puts the text into the input box.
 */
export async function transcribeAudio(blob: Blob): Promise<string> {
  const form = new FormData();
  form.append("audio", blob, "recording.webm");
  const data = await apiFetch<{ transcript: string }>("/chat/transcribe", {
    method: "POST",
    body: form,
  });
  return data.transcript;
}

/**
 * Send a voice recording to the campus assistant (full pipeline, kept for reference).
 */
export async function sendVoiceMessage(
  blob: Blob,
  sessionId?: string,
): Promise<VoiceResponse> {
  const form = new FormData();
  form.append("audio", blob, "recording.webm");
  if (sessionId) form.append("session_id", sessionId);

  return apiFetch<VoiceResponse>("/chat/voice", {
    method: "POST",
    body: form,
  });
}

/**
 * Send a campus image to the assistant (image-only query).
 */
export async function sendImageMessage(file: File): Promise<ImageResponse> {
  const form = new FormData();
  form.append("image", file);

  return apiFetch<ImageResponse>("/chat/image", {
    method: "POST",
    body: form,
  });
}

/**
 * Send a combined text + image query (Phase 4 multimodal).
 */
export async function sendMultimodalMessage(
  message: string,
  imageFile: File,
  sessionId?: string,
): Promise<MultimodalResponse> {
  const form = new FormData();
  form.append("message", message);
  form.append("image", imageFile);
  if (sessionId) form.append("session_id", sessionId);

  return apiFetch<MultimodalResponse>("/chat/multimodal", {
    method: "POST",
    body: form,
  });
}

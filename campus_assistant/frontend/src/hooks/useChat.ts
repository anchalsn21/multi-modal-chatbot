/**
 * useChat.ts — Custom React hook managing the full chat conversation state.
 *
 * Phase 4 additions:
 *   - session_id for conversation memory (UUID4 generated once per mount)
 *   - sendMultimodal: image + text fused query
 *   - isTranscribing: separate loading state for ASR phase of voice input
 */

"use client";

import { useState, useCallback, useRef } from "react";
import type { Message } from "@/types/chat";
import {
  streamTextMessage,
  sendImageMessage,
  sendMultimodalMessage,
} from "@/lib/api";
import { generateId } from "@/lib/utils";

function generateSessionId(): string {
  // Use crypto.randomUUID when available (all modern browsers), else fallback
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

interface UseChatReturn {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  sendText: (text: string) => Promise<void>;
  sendImage: (file: File, caption: string) => Promise<void>;
  sendMultimodal: (text: string, imageFile: File) => Promise<void>;
  clearMessages: () => void;
}

const WELCOME_MESSAGE: Message = {
  id: "welcome",
  role: "assistant",
  content:
    "👋 Hello! I'm your **Greenfield Campus Assistant**.\n\nI can help you find:\n- 📍 Campus locations\n- 🕘 Opening hours\n- 📅 Events\n- 🏛️ Departments\n- 📚 Study areas\n\nType a question below, or use the mic/image buttons to ask in other ways.",
  timestamp: new Date(),
  modality: "text",
};

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([WELCOME_MESSAGE]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionIdRef = useRef<string>(generateSessionId());

  const addMessage = useCallback((msg: Message) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  const sendText = useCallback(
    async (text: string) => {
      if (!text.trim()) return;

      addMessage({
        id: generateId(),
        role: "user",
        content: text.trim(),
        timestamp: new Date(),
        modality: "text",
      });
      setIsLoading(true);
      setError(null);

      // Create a placeholder bot message immediately — content grows token by token
      const botId = generateId();
      const botTimestamp = new Date();
      setMessages((prev) => [
        ...prev,
        {
          id: botId,
          role: "assistant",
          content: "",
          timestamp: botTimestamp,
          modality: "text" as const,
          streaming: true,
        },
      ]);

      try {
        await streamTextMessage(
          text.trim(),
          sessionIdRef.current,
          // onToken — append each word to the placeholder message
          (token: string) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === botId ? { ...m, content: m.content + token } : m
              )
            );
          },
          // onDone — attach intent/confidence metadata and clear streaming flag
          (meta) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === botId
                  ? { ...m, intent: meta.intent, confidence: meta.confidence, streaming: false }
                  : m
              )
            );
          }
        );
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Unknown error";
        setError(msg);
        // Replace the empty placeholder with the error message
        setMessages((prev) =>
          prev.map((m) =>
            m.id === botId ? { ...m, content: `⚠️ ${msg}` } : m
          )
        );
      } finally {
        setIsLoading(false);
      }
    },
    [addMessage]
  );

  const sendImage = useCallback(
    async (file: File, caption: string) => {
      const imageUrl = URL.createObjectURL(file);
      addMessage({
        id: generateId(),
        role: "user",
        content: caption || "",
        timestamp: new Date(),
        modality: "image",
        imageUrl,
        caption: caption || undefined,
      });
      setIsLoading(true);
      setError(null);

      try {
        const data = await sendImageMessage(file);
        addMessage({
          id: generateId(),
          role: "assistant",
          content: data.reply,
          timestamp: new Date(),
          modality: "image",
          confidence: data.confidence,
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Unknown error";
        setError(msg);
        addMessage({
          id: generateId(),
          role: "assistant",
          content: `⚠️ ${msg}`,
          timestamp: new Date(),
          modality: "text",
        });
      } finally {
        setIsLoading(false);
      }
    },
    [addMessage]
  );

  const sendMultimodal = useCallback(
    async (text: string, imageFile: File) => {
      const imageUrl = URL.createObjectURL(imageFile);
      addMessage({
        id: generateId(),
        role: "user",
        content: text,
        timestamp: new Date(),
        modality: "multimodal",
        imageUrl,
        caption: text,
      });
      setIsLoading(true);
      setError(null);

      try {
        const data = await sendMultimodalMessage(text, imageFile, sessionIdRef.current);
        addMessage({
          id: generateId(),
          role: "assistant",
          content: data.reply,
          timestamp: new Date(),
          modality: "multimodal",
          intent: data.intent,
          confidence: data.confidence,
          imageMatch: data.image_match ?? undefined,
          imageConfidence: data.image_confidence ?? undefined,
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Unknown error";
        setError(msg);
        addMessage({
          id: generateId(),
          role: "assistant",
          content: `⚠️ ${msg}`,
          timestamp: new Date(),
          modality: "text",
        });
      } finally {
        setIsLoading(false);
      }
    },
    [addMessage]
  );

  const clearMessages = useCallback(() => {
    setMessages([WELCOME_MESSAGE]);
    setError(null);
    // Generate a new session ID on clear to start fresh context
    sessionIdRef.current = generateSessionId();
  }, []);

  return {
    messages,
    isLoading,
    error,
    sendText,
    sendImage,
    sendMultimodal,
    clearMessages,
  };
}

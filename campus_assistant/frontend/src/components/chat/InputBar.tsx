"use client";

import { useState, useRef, useEffect, KeyboardEvent } from "react";
import { ArrowUp, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { VoiceButton } from "./VoiceButton";
import { ImageUpload } from "./ImageUpload";

interface InputBarProps {
  onSendText: (text: string) => void;
  onSendImage: (file: File, caption: string) => void;
  onSendMultimodal?: (text: string, imageFile: File) => void;
  isLoading: boolean;
}

export function InputBar({ onSendText, onSendImage, onSendMultimodal, isLoading }: InputBarProps) {
  const [text, setText] = useState("");
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [pendingPreview, setPendingPreview] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea whenever text changes (needed when transcript is injected)
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [text]);

  // Called by VoiceButton when Whisper finishes — populate input, focus it
  const handleTranscribed = (transcript: string) => {
    setText(transcript);
    setTimeout(() => {
      const el = textareaRef.current;
      if (!el) return;
      el.focus();
      el.setSelectionRange(el.value.length, el.value.length);
    }, 0);
  };

  const clearPending = () => {
    if (pendingPreview) URL.revokeObjectURL(pendingPreview);
    setPendingFile(null);
    setPendingPreview(null);
  };

  const handleSend = () => {
    if (isLoading) return;

    if (pendingFile) {
      const caption = text.trim();
      if (caption && onSendMultimodal) {
        // Both image and text present — use the multimodal fused pipeline
        onSendMultimodal(caption, pendingFile);
      } else {
        // Image only (or no multimodal handler)
        onSendImage(pendingFile, caption);
      }
      clearPending();
      setText("");
      if (textareaRef.current) textareaRef.current.style.height = "auto";
      return;
    }

    const trimmed = text.trim();
    if (!trimmed) return;
    onSendText(trimmed);
    setText("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  };

  const canSend = (!pendingFile ? text.trim().length > 0 : true) && !isLoading;

  return (
    <div className="border-t border-slate-200 bg-white px-4 py-4">
      <div className="mx-auto max-w-3xl space-y-2">

        {/* Staged image preview — shown above input until sent */}
        {pendingFile && pendingPreview && (
          <div className="flex items-start gap-2 rounded-xl border border-slate-200 bg-slate-50 p-2">
            <div className="relative flex-shrink-0">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={pendingPreview}
                alt="Attached image"
                className="h-16 w-16 rounded-lg object-cover border border-slate-200"
              />
              <button
                onClick={clearPending}
                className="absolute -top-1.5 -right-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-white border border-slate-300 hover:border-red-300 hover:bg-red-50 shadow-sm"
                title="Remove image"
              >
                <X className="h-3 w-3 text-slate-500" />
              </button>
            </div>
            <div className="flex flex-col justify-center min-w-0">
              <span className="text-xs font-medium text-slate-700 truncate">{pendingFile.name}</span>
              <span className="text-xs text-slate-400">Add a caption below (optional)</span>
            </div>
          </div>
        )}

        {/* Input container */}
        <div className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-white px-3 py-2.5 shadow-sm focus-within:border-indigo-400 focus-within:ring-2 focus-within:ring-indigo-100 transition-all">

          {/* Voice + image buttons */}
          <div className="flex items-center gap-1.5">
            <VoiceButton onTranscribed={handleTranscribed} disabled={isLoading} />
            <ImageUpload
              file={pendingFile}
              preview={pendingPreview}
              onFileSelected={(file, preview) => {
                setPendingFile(file);
                setPendingPreview(preview);
              }}
              onClear={clearPending}
              disabled={isLoading}
            />
          </div>

          {/* Textarea */}
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            onInput={handleInput}
            placeholder={pendingFile ? "Add a caption… (optional)" : "Ask about a campus location, hours, events… or use the mic 🎤"}
            rows={1}
            disabled={isLoading}
            className="flex-1 resize-none bg-transparent text-sm text-slate-900 placeholder-slate-400 focus:outline-none leading-relaxed disabled:opacity-50 self-center"
            style={{ maxHeight: "160px" }}
          />

          {/* Send button */}
          <button
            onClick={handleSend}
            disabled={!canSend}
            title="Send message (Enter)"
            className={cn(
              "flex-shrink-0 flex h-8 w-8 items-center justify-center rounded-xl transition-all",
              canSend
                ? "bg-indigo-600 text-white hover:bg-indigo-700 shadow-sm"
                : "bg-slate-100 text-slate-400 cursor-not-allowed"
            )}
          >
            <ArrowUp className="h-4 w-4" />
          </button>
        </div>

        <p className="text-center text-xs text-slate-400">
          Enter to send · Shift+Enter for newline · Upload image + type a question for multimodal search
        </p>
      </div>
    </div>
  );
}

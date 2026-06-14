"use client";

import { useRef, useState, useEffect } from "react";
import { User, Play, Pause } from "lucide-react";
import { cn, formatTime } from "@/lib/utils";
import type { Message } from "@/types/chat";

interface MessageBubbleProps {
  message: Message;
}

/**
 * Renders a single chat message bubble.
 *
 * User messages — right-aligned, indigo background, white text.
 * Bot messages  — left-aligned, white card with a subtle border and avatar.
 *
 * Basic **bold** markdown is converted to <strong> tags without a full
 * markdown library, keeping the dependency footprint small.
 */
export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={cn("flex items-end gap-3", isUser ? "flex-row-reverse" : "flex-row")}>

      {/* Avatar */}
      <div
        className={cn(
          "flex-shrink-0 flex h-8 w-8 items-center justify-center rounded-full text-xs font-bold",
          isUser
            ? "bg-indigo-600 text-white"
            : "bg-indigo-100 border border-indigo-200 text-indigo-600"
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : "AI"}
      </div>

      {/* Bubble */}
      <div
        className={cn(
          "max-w-[75%] rounded-2xl px-4 py-3 text-sm",
          isUser
            ? "rounded-br-sm bg-indigo-600 text-white shadow-sm"
            : "rounded-bl-sm bg-white border border-slate-200 text-slate-800 shadow-sm"
        )}
      >
        {/* WhatsApp-style audio player for voice messages */}
        {message.modality === "voice" && message.audioUrl && (
          <AudioPlayer url={message.audioUrl} isUser={isUser} />
        )}

        {/* Image for image/multimodal messages */}
        {(message.modality === "image" || message.modality === "multimodal") && message.imageUrl && (
          <div className="mb-2">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={message.imageUrl}
              alt={message.caption || "Sent image"}
              className="rounded-xl max-w-[220px] max-h-[200px] object-cover border border-white/20"
            />
          </div>
        )}

        {/* Message text — supports **bold** markdown; skip for voice user bubble (audio player covers it) */}
        {!(message.modality === "voice" && message.role === "user") &&
          (message.content || message.modality !== "image" || message.streaming) && (
          <div className="leading-relaxed whitespace-pre-wrap">
            {message.content
              ? <FormattedText content={message.content} isUser={isUser} />
              : message.streaming
              ? <span className="inline-block w-0.5 h-4 bg-indigo-400 animate-pulse align-middle" />
              : null
            }
            {message.streaming && message.content && (
              <span className="inline-block w-0.5 h-4 bg-indigo-400 animate-pulse align-middle ml-0.5" />
            )}
          </div>
        )}

        {/* Confidence progress bar — bot messages only */}
        {!isUser && message.confidence !== undefined && message.confidence > 0 && (
          <div className="mt-2">
            <div
              role="meter"
              aria-label={`Confidence: ${Math.round(message.confidence * 100)}%`}
              aria-valuenow={Math.round(message.confidence * 100)}
              aria-valuemin={0}
              aria-valuemax={100}
              className="h-1 w-full rounded-full bg-slate-100 overflow-hidden"
            >
              <div
                className={cn(
                  "h-full rounded-full transition-all duration-500",
                  message.confidence >= 0.8
                    ? "bg-green-400"
                    : message.confidence >= 0.5
                    ? "bg-yellow-400"
                    : "bg-red-400"
                )}
                style={{ width: `${message.confidence * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Timestamp + intent badge + image match badge */}
        <div
          className={cn(
            "mt-1.5 flex items-center flex-wrap gap-2 text-xs",
            isUser ? "text-indigo-200 justify-end" : "text-slate-400 justify-start"
          )}
        >
          <span>{formatTime(message.timestamp)}</span>
          {/* Intent badge — only on bot messages with a known intent */}
          {!isUser && message.intent && message.intent !== "unknown" && (
            <span className="rounded-full bg-indigo-50 border border-indigo-200 px-2 py-0.5 text-indigo-600">
              {message.intent.replace(/_/g, " ")}
            </span>
          )}
          {/* Image match badge — shown on multimodal responses */}
          {!isUser && message.imageMatch && (
            <span className="rounded-full bg-emerald-50 border border-emerald-200 px-2 py-0.5 text-emerald-700">
              📷 {message.imageMatch}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

/** WhatsApp-style audio player with play/pause and a live progress bar */
function AudioPlayer({ url, isUser }: { url: string; isUser: boolean }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    const onLoaded = () => setDuration(el.duration || 0);
    const onTime = () => {
      if (el.duration) setProgress(el.currentTime / el.duration);
    };
    const onEnded = () => { setPlaying(false); setProgress(0); };
    el.addEventListener("loadedmetadata", onLoaded);
    el.addEventListener("timeupdate", onTime);
    el.addEventListener("ended", onEnded);
    return () => {
      el.removeEventListener("loadedmetadata", onLoaded);
      el.removeEventListener("timeupdate", onTime);
      el.removeEventListener("ended", onEnded);
    };
  }, []);

  const toggle = () => {
    const el = audioRef.current;
    if (!el) return;
    if (playing) { el.pause(); setPlaying(false); }
    else { el.play(); setPlaying(true); }
  };

  const seek = (e: React.MouseEvent<HTMLDivElement>) => {
    const el = audioRef.current;
    if (!el) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    el.currentTime = ratio * el.duration;
    setProgress(ratio);
  };

  const fmtSec = (s: number) =>
    isFinite(s) ? `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}` : "0:00";

  return (
    <div className={cn("flex items-center gap-2 w-52 py-1", isUser ? "text-white" : "text-slate-700")}>
      <audio ref={audioRef} src={url} preload="metadata" />

      {/* Play / Pause button */}
      <button
        onClick={toggle}
        className={cn(
          "flex-shrink-0 flex h-8 w-8 items-center justify-center rounded-full transition-colors",
          isUser ? "bg-white/20 hover:bg-white/30" : "bg-indigo-100 hover:bg-indigo-200"
        )}
      >
        {playing
          ? <Pause className="h-4 w-4" />
          : <Play className="h-4 w-4 translate-x-[1px]" />}
      </button>

      {/* Progress bar + duration */}
      <div className="flex flex-col gap-1 flex-1 min-w-0">
        <div
          className={cn("h-1.5 rounded-full cursor-pointer", isUser ? "bg-white/30" : "bg-slate-200")}
          onClick={seek}
        >
          <div
            className={cn("h-full rounded-full transition-all", isUser ? "bg-white" : "bg-indigo-500")}
            style={{ width: `${progress * 100}%` }}
          />
        </div>
        <span className="text-xs opacity-70 tabular-nums">
          {fmtSec(duration > 0 ? progress * duration : 0)} / {fmtSec(duration)}
        </span>
      </div>
    </div>
  );
}

/**
 * Renders bot response markdown:
 * - First **bold** that appears alone on the opening line → styled heading
 * - All other **bold** → <strong>
 * - Plain text preserved with whitespace-pre-wrap on the parent
 */
function FormattedText({ content, isUser }: { content: string; isUser: boolean }) {
  // Split into lines so we can treat the first line specially for bot messages
  const lines = content.split("\n");
  const firstLine = lines[0].trim();
  const rest = lines.slice(1).join("\n");

  // Check if the entire first line is a single **heading**
  const headingMatch = !isUser && /^\*\*(.+)\*\*$/.test(firstLine);
  const headingText = headingMatch ? firstLine.slice(2, -2) : null;

  return (
    <>
      {headingText ? (
        <p className="text-base font-bold text-slate-900 mb-1">{headingText}</p>
      ) : (
        <InlineBold content={firstLine} isUser={isUser} />
      )}
      {rest && (
        <span>
          {"\n"}
          <InlineBold content={rest} isUser={isUser} />
        </span>
      )}
    </>
  );
}

/** Convert **bold** spans within a string to <strong> tags */
function InlineBold({ content, isUser }: { content: string; isUser: boolean }) {
  const parts = content.split(/\*\*(.*?)\*\*/g);
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <strong key={i} className={cn("font-semibold", isUser ? "text-white" : "text-slate-900")}>
            {part}
          </strong>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </>
  );
}

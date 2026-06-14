"use client";

import { useEffect, useRef } from "react";
import type { Message } from "@/types/chat";
import { MessageBubble } from "./MessageBubble";

interface ChatWindowProps {
  messages: Message[];
  isLoading: boolean;
  onQuickStart?: (text: string) => void;
}

const QUICK_STARTS = [
  "Where is the library?",
  "What time does the gym close?",
  "How do I get to Zone A?",
];

/**
 * Scrollable message list with:
 *   - Auto-scroll on new messages
 *   - Typing indicator while loading
 *   - "Transcribing…" label during ASR
 *   - Empty state with quick-start chips when only the welcome message is shown
 *   - aria-live for screen reader announcements
 */
export function ChatWindow({ messages, isLoading, onQuickStart }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Show quick-start chips only when the conversation has just the welcome message
  const showEmptyState = messages.length === 1 && messages[0].id === "welcome";

  return (
    <div className="flex-1 overflow-y-auto py-6">
      <div
        className="mx-auto w-full max-w-3xl px-4 space-y-4"
        aria-live="polite"
        aria-label="Chat conversation"
      >
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {/* Quick-start chips — shown only on empty conversation */}
        {showEmptyState && onQuickStart && (
          <div className="flex flex-wrap gap-2 pt-2">
            {QUICK_STARTS.map((chip) => (
              <button
                key={chip}
                onClick={() => onQuickStart(chip)}
                onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") onQuickStart(chip); }}
                className="rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1.5 text-xs text-indigo-700 hover:bg-indigo-100 transition-colors cursor-pointer"
              >
                {chip}
              </button>
            ))}
          </div>
        )}

        {/* Typing indicator */}
        {isLoading && (
          <div className="flex items-end gap-3" aria-label="Assistant is typing">
            <div className="flex-shrink-0 flex h-8 w-8 items-center justify-center rounded-full bg-indigo-100 border border-indigo-200">
              <span className="text-indigo-600 text-xs font-bold">AI</span>
            </div>
            <div className="rounded-2xl rounded-bl-sm bg-white border border-slate-200 shadow-sm px-4 py-3">
              <div className="flex items-center gap-1.5">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="h-2 w-2 rounded-full bg-indigo-400 animate-bounce"
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}

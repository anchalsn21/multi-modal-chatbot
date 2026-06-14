"use client";

import Link from "next/link";
import { GraduationCap, ArrowLeft, Trash2, MapPin, Clock, CalendarDays, Building2, BookOpen, Navigation } from "lucide-react";
import { useChat } from "@/hooks/useChat";
import { ChatWindow } from "@/components/chat/ChatWindow";
import { InputBar } from "@/components/chat/InputBar";
import { CampusMap } from "@/components/chat/CampusMap";

/** Suggested starter prompts shown in the sidebar */
const SUGGESTED_PROMPTS = [
  { icon: MapPin,      text: "Where is the main library?" },
  { icon: Clock,       text: "What time does the gym close?" },
  { icon: CalendarDays,text: "Tell me about the freshers fair" },
  { icon: Building2,   text: "Where is the CS department?" },
  { icon: BookOpen,    text: "Is there a 24-hour study room?" },
  { icon: Navigation,  text: "How do I get from A to K?" },
];

/**
 * Chat page — full-height layout with sidebar + chat window.
 *
 * Layout:
 *   - Left sidebar (desktop only): logo, suggested prompts, scope badge
 *   - Right main: ChatWindow (scrollable) + InputBar (sticky bottom)
 */
export default function ChatPage() {
  const { messages, isLoading, sendText, sendImage, sendMultimodal, clearMessages } = useChat();

  return (
    <div className="flex h-[100dvh] bg-slate-50 text-slate-900 overflow-hidden">

      {/* ── Sidebar ──────────────────────────────────────────────────── */}
      <aside className="hidden lg:flex flex-col w-64 border-r border-slate-200 bg-white flex-shrink-0">

        {/* Logo */}
        <div className="flex items-center gap-2.5 px-5 py-5 border-b border-slate-100">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600">
            <GraduationCap className="h-3.5 w-3.5 text-white" />
          </div>
          <span className="text-sm font-semibold text-slate-900">
            Greenfield <span className="text-indigo-600">Campus AI</span>
          </span>
        </div>

        {/* Suggested prompts */}
        <div className="flex-1 overflow-y-auto px-3 py-4">
          <p className="mb-2 px-2 text-xs font-semibold uppercase tracking-widest text-slate-400">
            Try asking
          </p>
          <div className="space-y-0.5">
            {SUGGESTED_PROMPTS.map(({ icon: Icon, text }) => (
              <button
                key={text}
                onClick={() => sendText(text)}
                disabled={isLoading}
                className="w-full flex items-center gap-3 rounded-lg px-3 py-2.5 text-left text-xs text-slate-600 hover:bg-slate-50 hover:text-slate-900 transition-colors disabled:opacity-40"
              >
                <Icon className="h-3.5 w-3.5 flex-shrink-0 text-indigo-500" />
                <span className="truncate">{text}</span>
              </button>
            ))}
          </div>

          {/* Map hint card */}
          <div className="mt-5 rounded-xl border border-indigo-100 bg-indigo-50 p-3">
            <p className="text-xs font-semibold text-indigo-700 mb-1">📍 Campus Map</p>
            <p className="text-xs text-indigo-600/80">
              Zones are labelled <strong>A–P</strong>. Open the map below, then ask for directions by zone.
            </p>
          </div>
        </div>

        {/* Footer actions */}
        <div className="border-t border-slate-100 p-3 space-y-0.5">
          {/* Campus map button — opens modal overlay */}
          <CampusMap />
          <button
            onClick={clearMessages}
            className="w-full flex items-center gap-2 rounded-lg px-3 py-2.5 text-xs text-slate-500 hover:bg-slate-50 hover:text-slate-900 transition-colors"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Clear conversation
          </button>
          <Link
            href="/"
            className="flex items-center gap-2 rounded-lg px-3 py-2.5 text-xs text-slate-500 hover:bg-slate-50 hover:text-slate-900 transition-colors"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Back to home
          </Link>
        </div>
      </aside>

      {/* ── Main chat area ──────────────────────────────────────────── */}
      <main className="flex flex-1 flex-col min-w-0 bg-slate-50">

        {/* Mobile header */}
        <header className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-3 lg:hidden">
          <Link href="/" className="flex items-center gap-1.5 text-slate-500 hover:text-slate-900 transition-colors">
            <ArrowLeft className="h-4 w-4" />
            <span className="text-xs">Home</span>
          </Link>
          <div className="flex items-center gap-2">
            <GraduationCap className="h-4 w-4 text-indigo-600" />
            <span className="text-sm font-semibold text-slate-900">Campus AI</span>
          </div>
          <div className="flex items-center gap-1">
            {/* Mobile map button — renders its own modal */}
            <CampusMap variant="icon" />
            <button
              onClick={clearMessages}
              className="text-slate-400 hover:text-slate-700 transition-colors"
              title="Clear messages"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          </div>
        </header>

        {/* Messages */}
        <ChatWindow
          messages={messages}
          isLoading={isLoading}
          onQuickStart={sendText}
        />

        {/* Input */}
        <InputBar
          onSendText={sendText}
          onSendImage={(file, caption) => sendImage(file, caption)}
          onSendMultimodal={sendMultimodal}
          isLoading={isLoading}
        />
      </main>
    </div>
  );
}

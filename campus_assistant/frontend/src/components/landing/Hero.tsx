import Link from "next/link";
import { ArrowRight, ChevronDown, Mic, ImageIcon, MessageSquare } from "lucide-react";

export function Hero() {
  return (
    <section className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden bg-white px-6 text-center pt-16">
      {/* Subtle background pattern */}
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.025]"
        aria-hidden="true"
        style={{
          backgroundImage:
            "radial-gradient(circle, #4f46e5 1px, transparent 1px)",
          backgroundSize: "32px 32px",
        }}
      />

      {/* Soft colour wash */}
      <div
        className="pointer-events-none absolute top-0 left-0 right-0 h-[60%] opacity-30"
        aria-hidden="true"
        style={{
          background:
            "radial-gradient(ellipse 80% 50% at 50% -10%, #e0e7ff, transparent)",
        }}
      />

      {/* Status badge */}
      <div className="relative mb-6 inline-flex items-center gap-2 rounded-full border border-indigo-200 bg-indigo-50 px-4 py-1.5 text-xs font-medium text-indigo-700">
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-indigo-400 opacity-75" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-indigo-500" />
        </span>
        Multimodal AI — Text, Voice &amp; Image
      </div>

      {/* Headline */}
      <h1 className="relative max-w-3xl text-5xl font-bold leading-tight tracking-tight text-slate-900 sm:text-6xl lg:text-7xl">
        Your{" "}
        <span className="text-indigo-600">AI-Powered</span>
        <br />
        Campus Guide
      </h1>

      {/* Subtext */}
      <p className="relative mt-6 max-w-xl text-lg text-slate-500">
        Find locations, check opening hours, discover events, and navigate Greenfield University
        campus — all through a single intelligent assistant.
      </p>

      {/* Modality pills */}
      <div className="relative mt-8 flex flex-wrap items-center justify-center gap-3">
        {[
          { icon: MessageSquare, label: "Text Queries",  color: "text-indigo-600",  bg: "bg-indigo-50  border-indigo-200"  },
          { icon: Mic,           label: "Voice Input",   color: "text-sky-600",     bg: "bg-sky-50     border-sky-200"     },
          { icon: ImageIcon,     label: "Image Upload",  color: "text-emerald-600", bg: "bg-emerald-50 border-emerald-200" },
        ].map(({ icon: Icon, label, color, bg }) => (
          <span
            key={label}
            className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium ${bg} ${color}`}
          >
            <Icon className="h-3 w-3" />
            {label}
          </span>
        ))}
      </div>

      {/* CTA buttons */}
      <div className="relative mt-10 flex flex-col items-center gap-3 sm:flex-row sm:justify-center">
        <Link
          href="/chat"
          className="flex items-center gap-2 rounded-xl bg-indigo-600 px-8 py-3.5 text-sm font-semibold text-white shadow-md hover:bg-indigo-700 transition-colors"
        >
          Start Chatting
          <ArrowRight className="h-4 w-4" />
        </Link>
        <a
          href="#features"
          className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-8 py-3.5 text-sm font-medium text-slate-600 hover:border-slate-300 hover:text-slate-900 transition-colors"
        >
          Explore Features
          <ChevronDown className="h-4 w-4" />
        </a>
      </div>

      {/* Scroll indicator */}
      <div className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-slate-400">
        <span className="text-xs">Scroll to explore</span>
        <ChevronDown className="h-4 w-4 animate-bounce" />
      </div>
    </section>
  );
}

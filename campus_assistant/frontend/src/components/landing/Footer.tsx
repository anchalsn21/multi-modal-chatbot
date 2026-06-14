import { GraduationCap } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-slate-200 bg-white py-10 px-6">
      <div className="mx-auto max-w-6xl">
        <div className="flex flex-col items-center gap-4 text-center sm:flex-row sm:justify-between sm:text-left">
          {/* Brand */}
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600">
              <GraduationCap className="h-3.5 w-3.5 text-white" />
            </div>
            <span className="text-sm font-semibold text-slate-900">
              Greenfield <span className="text-indigo-600">Campus AI</span>
            </span>
          </div>

          {/* Info */}
          <p className="text-xs text-slate-400">
            MSc AI Assignment. Built with DistilBERT, CLIP, Whisper, FastAPI, and Next.js.
          </p>
        </div>
      </div>
    </footer>
  );
}

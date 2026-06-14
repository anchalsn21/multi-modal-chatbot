"use client";

import { useState } from "react";
import { Map, X, ZoomIn, ZoomOut, Maximize2 } from "lucide-react";

/**
 * CampusMap — a modal that displays the Greenfield campus map SVG.
 *
 * The trigger is a small button passed as children or rendered inline.
 * Zoom is handled with CSS transform so no extra library is needed.
 */
interface CampusMapProps {
  /** "sidebar" (default) = full-width text button; "icon" = compact icon-only button for mobile */
  variant?: "sidebar" | "icon";
}

export function CampusMap({ variant = "sidebar" }: CampusMapProps) {
  const [open, setOpen] = useState(false);
  const [zoom, setZoom] = useState(1);

  const zoomIn  = () => setZoom((z) => Math.min(z + 0.25, 3));
  const zoomOut = () => setZoom((z) => Math.max(z - 0.25, 0.5));
  const reset   = () => setZoom(1);

  return (
    <>
      {/* ── Trigger button ───────────────────────────────────────────── */}
      {variant === "icon" ? (
        <button
          onClick={() => { setOpen(true); setZoom(1); }}
          className="rounded-lg p-1.5 text-slate-400 hover:bg-indigo-50 hover:text-indigo-600 transition-colors"
          title="Campus Map"
        >
          <Map className="h-4 w-4" />
        </button>
      ) : (
        <button
          onClick={() => { setOpen(true); setZoom(1); }}
          className="w-full flex items-center gap-2 rounded-lg px-3 py-2.5 text-xs text-slate-500 hover:bg-indigo-50 hover:text-indigo-700 transition-colors"
        >
          <Map className="h-3.5 w-3.5 flex-shrink-0" />
          Campus Map
        </button>
      )}

      {/* ── Modal overlay ────────────────────────────────────────────── */}
      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
          onClick={() => setOpen(false)}
        >
          <div
            className="relative flex flex-col bg-white rounded-2xl shadow-2xl overflow-hidden w-full max-w-5xl max-h-[90vh]"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-3 border-b border-slate-100 flex-shrink-0">
              <div className="flex items-center gap-2">
                <Map className="h-4 w-4 text-indigo-600" />
                <span className="text-sm font-semibold text-slate-800">
                  Greenfield University — Campus Map
                </span>
                <span className="ml-2 rounded-full bg-indigo-50 px-2 py-0.5 text-xs text-indigo-600 font-medium">
                  5 Zones · 13 Locations
                </span>
              </div>
              <button
                onClick={() => setOpen(false)}
                className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700 transition-colors"
                title="Close"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Map image — scrollable so it stays usable when zoomed */}
            <div className="flex-1 overflow-auto bg-slate-50 flex items-center justify-center p-4">
              <img
                src="/campus_map.svg"
                alt="Greenfield University Campus Map showing zones A through P"
                className="origin-center transition-transform duration-200 rounded-lg shadow"
                style={{ transform: `scale(${zoom})`, transformOrigin: "top center" }}
                draggable={false}
              />
            </div>

            {/* Footer — zoom controls + hint */}
            <div className="flex items-center justify-between px-5 py-2.5 border-t border-slate-100 bg-white flex-shrink-0">
              <p className="text-xs text-slate-400">
                Tip: ask the assistant <span className="font-medium text-slate-600">"How do I get from A to K?"</span> for turn-by-turn directions.
              </p>
              <div className="flex items-center gap-1">
                <button
                  onClick={zoomOut}
                  disabled={zoom <= 0.5}
                  className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 disabled:opacity-30 transition-colors"
                  title="Zoom out"
                >
                  <ZoomOut className="h-4 w-4" />
                </button>
                <button
                  onClick={reset}
                  className="rounded px-2 py-1 text-xs text-slate-500 hover:bg-slate-100 transition-colors min-w-[3rem] text-center"
                  title="Reset zoom"
                >
                  {Math.round(zoom * 100)}%
                </button>
                <button
                  onClick={zoomIn}
                  disabled={zoom >= 3}
                  className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 disabled:opacity-30 transition-colors"
                  title="Zoom in"
                >
                  <ZoomIn className="h-4 w-4" />
                </button>
                <button
                  onClick={reset}
                  className="ml-1 rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 transition-colors"
                  title="Fit to screen"
                >
                  <Maximize2 className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

"use client";

import { useRef } from "react";
import { ImageIcon, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface ImageUploadProps {
  /** Currently staged file (null = nothing staged) */
  file: File | null;
  /** Preview object URL for the staged file */
  preview: string | null;
  onFileSelected: (file: File, preview: string) => void;
  onClear: () => void;
  disabled?: boolean;
}

export function ImageUpload({ file, preview, onFileSelected, onClear, disabled }: ImageUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (!selected) return;
    const objectUrl = URL.createObjectURL(selected);
    onFileSelected(selected, objectUrl);
    e.target.value = "";
  };

  return (
    <div className="flex items-center gap-1.5">
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileChange}
        disabled={disabled}
      />

      {/* Upload button — hidden once a file is staged */}
      {!file && (
        <button
          onClick={() => inputRef.current?.click()}
          disabled={disabled}
          title="Attach an image"
          className={cn(
            "flex h-8 w-8 items-center justify-center rounded-xl border transition-all",
            "bg-slate-50 text-slate-500 border-slate-200 hover:border-indigo-300 hover:text-indigo-600",
            disabled && "opacity-40 cursor-not-allowed"
          )}
        >
          <ImageIcon className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
}

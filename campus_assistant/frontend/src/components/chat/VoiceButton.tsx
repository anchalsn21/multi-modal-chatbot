"use client";

import { useState, useRef, useCallback } from "react";
import { Mic, MicOff, Square, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { transcribeAudio } from "@/lib/api";

interface VoiceButtonProps {
  /** Called with the transcribed text once Whisper finishes. */
  onTranscribed: (text: string) => void;
  disabled?: boolean;
}

/**
 * Microphone button that records audio, sends it to /chat/transcribe,
 * and passes the transcript back to the parent via onTranscribed().
 *
 * The parent puts the transcript into the text input so the user can
 * review and edit before sending — no automatic submission.
 *
 * States:
 *   idle         — mic icon, ready to record
 *   recording    — red pulsing stop square
 *   transcribing — spinner while Whisper runs
 *   error        — mic-off icon with tooltip
 */
export function VoiceButton({ onTranscribed, disabled }: VoiceButtonProps) {
  const [isRecording, setIsRecording]       = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [error, setError]                   = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef        = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    setError(null);
    try {
      const stream   = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setIsTranscribing(true);
        try {
          const text = await transcribeAudio(blob);
          onTranscribed(text);
        } catch {
          setError("Transcription failed. Please try again.");
        } finally {
          setIsTranscribing(false);
        }
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    } catch {
      setError("Microphone access denied. Please allow microphone permissions.");
    }
  }, [onTranscribed]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  const isDisabled = (disabled && !isRecording) || isTranscribing;

  return (
    <div className="relative">
      <button
        onClick={isRecording ? stopRecording : startRecording}
        disabled={isDisabled}
        aria-label={
          isTranscribing ? "Transcribing audio"
          : isRecording   ? "Stop recording"
          :                 "Record voice message"
        }
        title={
          isTranscribing ? "Transcribing…"
          : isRecording   ? "Stop recording"
          :                 "Record voice message"
        }
        className={cn(
          "flex h-8 w-8 items-center justify-center rounded-xl border transition-all",
          isRecording
            ? "bg-red-50 text-red-500 border-red-200 animate-pulse"
            : isTranscribing
            ? "bg-amber-50 text-amber-500 border-amber-200"
            : "bg-slate-50 text-slate-500 border-slate-200 hover:border-indigo-300 hover:text-indigo-600",
          isDisabled && "opacity-40 cursor-not-allowed"
        )}
      >
        {isTranscribing ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : isRecording ? (
          <Square className="h-3.5 w-3.5" />
        ) : error ? (
          <MicOff className="h-3.5 w-3.5 text-red-400" />
        ) : (
          <Mic className="h-3.5 w-3.5" />
        )}
      </button>

      {error && (
        <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 w-52 rounded-xl bg-white border border-red-200 shadow-lg p-2.5 text-xs text-red-600 text-center">
          {error}
        </div>
      )}

      {isTranscribing && (
        <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 w-36 rounded-xl bg-white border border-amber-200 shadow-lg p-2 text-xs text-amber-700 text-center">
          Transcribing…
        </div>
      )}
    </div>
  );
}

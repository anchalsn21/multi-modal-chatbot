/**
 * utils.ts — Shared utility functions.
 *
 * cn() is the standard shadcn/ui helper for merging Tailwind class names.
 * It combines clsx (conditional class logic) with tailwind-merge
 * (deduplication of conflicting Tailwind classes).
 *
 * Usage:
 *   cn("px-4 py-2", isActive && "bg-violet-600", className)
 */

import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Format a Date to a short time string, e.g. "14:32" */
export function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/** Generate a random UUID-like string for message IDs */
export function generateId(): string {
  return Math.random().toString(36).slice(2, 11);
}

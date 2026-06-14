"use client";

import Link from "next/link";
import { GraduationCap } from "lucide-react";

export function Navbar() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 border-b border-slate-200 bg-white/90 backdrop-blur-sm">
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 shadow-sm group-hover:bg-indigo-700 transition-colors">
            <GraduationCap className="h-4 w-4 text-white" />
          </div>
          <span className="text-sm font-semibold text-slate-900">
            Greenfield <span className="text-indigo-600">Campus AI</span>
          </span>
        </Link>

        {/* Nav links */}
        <div className="hidden items-center gap-6 text-sm text-slate-500 md:flex">
          <a href="#features" className="hover:text-slate-900 transition-colors">
            Features
          </a>
        </div>

        {/* CTA */}
        <Link
          href="/chat"
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 transition-colors"
        >
          Launch Assistant
        </Link>
      </nav>
    </header>
  );
}

import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });

export const metadata: Metadata = {
  title: "Greenfield Campus Assistant — AI-Powered Campus Guide",
  description:
    "Your intelligent campus orientation assistant. Find locations, check hours, discover events, and explore departments at Greenfield University.",
  keywords: ["campus", "Greenfield", "AI assistant", "chatbot", "university", "orientation"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body>{children}</body>
    </html>
  );
}

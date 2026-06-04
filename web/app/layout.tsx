import type { Metadata } from "next";
import { Fraunces, Hanken_Grotesk, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const fraunces = Fraunces({ subsets: ["latin"], variable: "--font-display", display: "swap" });
const hanken = Hanken_Grotesk({ subsets: ["latin"], variable: "--font-ui", display: "swap" });
const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Glass-Box Video Analyst",
  description: "An LLM writes a visual program; watch it run over a video, step by step.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${fraunces.variable} ${hanken.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}

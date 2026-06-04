import type { ProgramStep } from "./types";

// Render the JSON DSL as Python-like code tokens for the PROGRAM panel. Deterministic
// (we build the tokens, no parsing), so coloring is reliable. A lightweight stand-in for
// the locked Shiki/Prism (D5) until that's wired in.

export type Tok = { c: "kw" | "fn" | "str" | "num" | "op" | "name" | ""; t: string };

const nm = (t: string): Tok => ({ c: "name", t });
const fn = (t: string): Tok => ({ c: "fn", t });
const op = (t: string): Tok => ({ c: "op", t });
const num = (t: unknown): Tok => ({ c: "num", t: String(t) });
const str = (t: string): Tok => ({ c: "str", t });
const kw = (t: string): Tok => ({ c: "kw", t });

function call(id: string, fnName: string, groups: Tok[][]): Tok[] {
  const out: Tok[] = [nm(id), op(" = "), fn(fnName), op("(")];
  groups.forEach((g, i) => {
    if (i) out.push(op(", "));
    out.push(...g);
  });
  out.push(op(")"));
  return out;
}

export const COMMENTS: Record<string, string> = {
  sample_frames: "# sample frames around the goal moment",
  detect: "# find every player on the pitch",
  crop: "# isolate each jersey-number region",
  read_text: "# OCR the number off each shirt",
  filter: "# keep only player #10",
  count: "# how many?",
  temporal_order: "# order events — was it the FIRST goal?",
  answer: "# answer the question from the evidence",
};

function lineFor(s: ProgramStep): Tok[] {
  const a = s.args as Record<string, any>;
  switch (s.op) {
    case "sample_frames":
      return call(s.id, "sample_frames", [[num(a.start_ms)], [num(a.end_ms)], [kw("fps"), op("="), num(a.fps)]]);
    case "detect":
      return call(s.id, "detect", [[nm(a.frames)], [op("["), str(`"${(a.classes || []).join('", "')}"`), op("]")]]);
    case "crop":
      return call(s.id, "crop", [[nm(a.detections)], [str(`"${a.region}"`)]]);
    case "read_text":
      return call(s.id, "read_text", [[nm(a.crops)]]);
    case "filter":
      return call(s.id, "filter", [[nm(a.items)], [nm(a.where.field), op(" == "), str(`"${a.where.equals}"`)]]);
    case "count":
      return call(s.id, "count", [[nm(a.items)]]);
    case "temporal_order":
      return call(s.id, "temporal_order", [[nm(a.events)], [kw("by"), op("="), str(`"${a.by}"`)]]);
    case "answer":
      return call(s.id, "answer", [[nm(a.from)], [str(`"${a.question}"`)]]);
    default:
      return [nm(s.id), op(" = "), fn(s.op), op("(…)")];
  }
}

export function programToLines(program: ProgramStep[]): Tok[][] {
  return program.map(lineFor);
}

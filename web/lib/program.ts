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

// The DSL args are validated upstream (api/schema/run_doc.schema.json); here we only read fields
// off a step's already-trusted args by op. A loose recursive shape lets us index nested fields
// (e.g. a.where.field) and array methods (a.classes?.join) without reaching for `any`.
type Args = { [key: string]: ArgValue };
type ArgValue = string | number | boolean | null | undefined | ArgValue[] | Args;

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
  describe_scene: "# describe what's happening in the scene",
  crop: "# isolate each jersey-number region",
  read_text: "# OCR the number off each shirt",
  filter: "# keep only player #10",
  count: "# how many?",
  temporal_order: "# order events — was it the FIRST goal?",
  answer: "# answer the question from the evidence",
};

function lineFor(s: ProgramStep): Tok[] {
  const a = s.args as Args;
  switch (s.op) {
    case "sample_frames":
      return call(s.id, "sample_frames", [[num(a.start_ms)], [num(a.end_ms)], [kw("fps"), op("="), num(a.fps)]]);
    case "detect":
      return call(s.id, "detect", [[nm(String(a.frames))], [op("["), str(`"${((a.classes as string[]) || []).join('", "')}"`), op("]")]]);
    case "describe_scene": {
      const groups: Tok[][] = [[nm(String(a.frames))]];
      if (a.max_captions != null) groups.push([kw("max_captions"), op("="), num(a.max_captions)]);
      return call(s.id, "describe_scene", groups);
    }
    case "crop":
      return call(s.id, "crop", [[nm(String(a.detections))], [str(`"${a.region}"`)]]);
    case "read_text":
      return call(s.id, "read_text", [[nm(String(a.crops))]]);
    case "filter": {
      const where = a.where as Args;
      return call(s.id, "filter", [[nm(String(a.items))], [nm(String(where.field)), op(" == "), str(`"${where.equals}"`)]]);
    }
    case "count":
      return call(s.id, "count", [[nm(String(a.items))]]);
    case "temporal_order":
      return call(s.id, "temporal_order", [[nm(String(a.events))], [kw("by"), op("="), str(`"${a.by}"`)]]);
    case "answer":
      return call(s.id, "answer", [[nm(String(a.from))], [str(`"${a.question}"`)]]);
    default:
      return [nm(s.id), op(" = "), fn(s.op), op("(…)")];
  }
}

export function programToLines(program: ProgramStep[]): Tok[][] {
  return program.map(lineFor);
}

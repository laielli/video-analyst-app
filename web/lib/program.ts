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

// Per-step comment for the PROGRAM panel, derived ONLY from this step's own (already-trusted)
// args — never from the question text, the clip, or which canned demo this resembles. The panel
// is captioned "AI reasoning", so a comment that guesses at INTENT ("around the goal moment",
// "keep only player #10") is a lie the moment the same op appears in a different program (a
// free-text "at the start of the video?" query, or a filter on #7/#23) — this happened in
// production. Comments here describe MECHANICALLY what the step computes, which is true for
// every program that could ever produce this op with these args.
function sampledFrameCount(startMs: number, endMs: number, fps: number): number {
  // Mirrors interpreter/primitives.py:op_sample_frames's stride math exactly (len(range(start,
  // end+1, stride)) with stride = max(1, round(1000/fps))) so the count quoted here is never
  // off from what the step actually samples.
  const stride = Math.max(1, Math.round(1000 / fps));
  return Math.floor((endMs - startMs) / stride) + 1;
}

export function commentFor(s: ProgramStep): string {
  const a = s.args as Args;
  switch (s.op) {
    case "sample_frames": {
      const n = sampledFrameCount(Number(a.start_ms), Number(a.end_ms), Number(a.fps));
      return n <= 1 ? "# sample a single frame" : `# sample ${n} frames across the window`;
    }
    case "detect": {
      const cls = ((a.classes as string[]) || []).join(", ") || "objects";
      const onPitch = a.on_pitch ? ", keeping only on-pitch boxes" : "";
      return `# detect ${cls} in each frame${onPitch}`;
    }
    case "describe_scene":
      return "# caption each sampled frame";
    case "crop":
      return `# crop the "${a.region}" region from each detection`;
    case "read_text":
      return "# read text out of each crop";
    case "filter": {
      const where = a.where as Args;
      return `# keep items where ${where.field} == "${where.equals}"`;
    }
    case "count":
      return "# count the items";
    case "temporal_order":
      return `# order events by ${a.by}`;
    case "answer":
      return "# answer the question from the evidence";
    default:
      return `# ${s.op}`;
  }
}

function lineFor(s: ProgramStep): Tok[] {
  const a = s.args as Args;
  switch (s.op) {
    case "sample_frames":
      return call(s.id, "sample_frames", [[num(a.start_ms)], [num(a.end_ms)], [kw("fps"), op("="), num(a.fps)]]);
    case "detect": {
      const groups: Tok[][] = [[nm(String(a.frames))], [op("["), str(`"${((a.classes as string[]) || []).join('", "')}"`), op("]")]];
      // on_pitch is the crowd-filter arg — surfacing it in the rendered program is the point
      // (the glass box shows HOW spectators/staff get excluded, not just that they are).
      if (a.on_pitch != null) groups.push([kw("on_pitch"), op("="), kw(a.on_pitch ? "True" : "False")]);
      return call(s.id, "detect", groups);
    }
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

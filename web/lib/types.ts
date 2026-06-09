// Mirrors api/schema/run_doc.schema.json — the contract the backend streams.

export type Box = { x: number; y: number; w: number; h: number };
export type Tone = "azure" | "teal" | "green" | "amber" | "dim";

export type Overlay = {
  box: Box;
  label?: string;
  confidence?: number | null;
  tone: Tone;
  kind?: "box" | "crop" | "goal" | "ball";
};

export type StepStatus = "pending" | "active" | "done" | "empty" | "error";
export type Source = "live" | "cached" | "pinned";
// Provenance of the PROGRAM (Q1 -> A): "ungrounded" = a free-text query that couldn't be
// grounded and has no pinned fallback — the honest "couldn't ground this" state.
export type ProgramSource = "live" | "pinned" | "ungrounded";

export type StepResult = {
  id: string;
  op: string;
  producer?: string;
  status: StepStatus;
  source: Source;
  inputs?: string[];
  input_label?: string;
  output_label?: string;
  confidence?: number | null;
  evidence?: { frame_ts_ms: number; overlays: Overlay[] };
  note?: string;
};

export type ProgramStep = {
  id: string;
  op: string;
  args: Record<string, unknown>;
};

export type Clip = { id: string; width: number; height: number; duration_ms: number; fps?: number };

export type Findings = {
  // null when grounded === false (the honest "couldn't ground this" state, Q1 -> A).
  answer: string | null;
  verdict: string | null;
  // The ungrounded discriminator: false => no answer for the question asked; reason is a fixed
  // enum (e.g. "codegen-disabled", "no-grounded-answer"). A grounded run has grounded:true,
  // reason:null. May be absent on legacy/canned docs — treat missing as grounded.
  grounded?: boolean;
  reason?: string | null;
  supporting_step?: string;
  partial?: boolean;
  partial_note?: string;
};

export type Meta = {
  query: string;
  clip: Clip;
  program: ProgramStep[];
  total_steps: number;
  pace_ms: number;
  // Provenance of the program: "live" = compiled this run by Azure OpenAI codegen and
  // validated; "pinned" = the known-good fallback (no creds, or live failed — D-DR6);
  // "ungrounded" = a free-text query that couldn't be grounded (no pinned fallback).
  program_source?: ProgramSource;
  // Permalink fields (free-text cache + shareable replays). `run_id` is the stable
  // content-address present only for a grounded free-text run or a permalink replay (null
  // otherwise) — the "Copy run link" affordance is enabled iff it's set. `cached` is true when
  // the doc was served from the store (a free-text cache hit or a permalink replay).
  run_id?: string | null;
  cached?: boolean;
};

export type CatalogQuery = { id: string; text: string; clip: string };

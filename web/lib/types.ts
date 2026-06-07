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

// Q1 -> A: an ungrounded free-text run carries grounded:false + a fixed reason; answer/verdict
// are null (NOT a fabricated string). A grounded run carries the strings + grounded:true.
export type UngroundedReason =
  | "codegen-disabled"
  | "codegen-error"
  | "invalid-program"
  | "execution-error"
  | "no-grounded-answer";

export type Findings = {
  answer: string | null;
  verdict: string | null;
  grounded?: boolean;
  reason?: UngroundedReason | null;
  supporting_step?: string;
  partial?: boolean;
  partial_note?: string;
};

export type ProgramSource = "live" | "pinned" | "ungrounded";

export type Meta = {
  query: string;
  clip: Clip;
  program: ProgramStep[];
  total_steps: number;
  pace_ms: number;
  // Provenance of the program: "live" = compiled this run by Azure OpenAI codegen and
  // validated; "pinned" = the known-good fallback (no creds, or live failed — D-DR6);
  // "ungrounded" = a free-text query with no pinned fallback that couldn't be grounded.
  program_source?: ProgramSource;
};

export type CatalogQuery = { id: string; text: string; clip: string };

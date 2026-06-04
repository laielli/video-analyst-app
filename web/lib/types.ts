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

export type Findings = {
  answer: string;
  verdict: string;
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
};

export type CatalogQuery = { id: string; text: string; clip: string };

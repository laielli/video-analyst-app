import { describe, it, expect } from "vitest";
import { programToLines, commentFor } from "@/lib/program";
import type { ProgramStep } from "@/lib/types";

// program.ts is pure (it builds colored tokens from the DSL, no parsing). We assert one case per
// branch in lineFor: sample_frames, detect, describe_scene, filter, answer, and the default
// fallback — and that commentFor derives an honest, args-driven comment for every op.

function tokens(line: { c: string; t: string }[]) {
  return line.map((tok) => tok.t).join("");
}

describe("programToLines", () => {
  it("renders sample_frames with start/end/fps", () => {
    const step: ProgramStep = { id: "frames", op: "sample_frames", args: { start_ms: 3500, end_ms: 5000, fps: 8 } };
    const [line] = programToLines([step]);
    const s = tokens(line);
    expect(s).toContain("frames = sample_frames(");
    expect(s).toContain("3500");
    expect(s).toContain("5000");
    expect(s).toContain("fps=8");
  });

  it("renders detect with classes joined", () => {
    const step: ProgramStep = { id: "people", op: "detect", args: { frames: "frames", classes: ["person", "ball"] } };
    const [line] = programToLines([step]);
    const s = tokens(line);
    expect(s).toContain("people = detect(");
    expect(s).toContain("frames");
    expect(s).toContain('"person", "ball"');
    // no on_pitch arg -> no on_pitch token (the arg is optional and off by default)
    expect(s).not.toContain("on_pitch");
  });

  it("renders detect's on_pitch crowd-filter arg when present", () => {
    // Surfacing on_pitch in the rendered program is the point of the arg — the glass box
    // shows HOW spectators/staff get excluded from "person" detections, not just that they are.
    const step: ProgramStep = {
      id: "people", op: "detect",
      args: { frames: "frames", classes: ["person"], on_pitch: true },
    };
    const [line] = programToLines([step]);
    const s = tokens(line);
    expect(s).toContain("on_pitch=True");
  });

  it("renders describe_scene with frames (and optional max_captions)", () => {
    const bare: ProgramStep = { id: "scene", op: "describe_scene", args: { frames: "frames" } };
    const [line] = programToLines([bare]);
    const s = tokens(line);
    expect(s).toContain("scene = describe_scene(");
    expect(s).toContain("frames");
    expect(s).not.toContain("max_captions");

    const capped: ProgramStep = { id: "scene", op: "describe_scene", args: { frames: "frames", max_captions: 4 } };
    const [line2] = programToLines([capped]);
    const s2 = tokens(line2);
    expect(s2).toContain("max_captions=4");
  });

  it("renders filter with where.field == equals", () => {
    const step: ProgramStep = {
      id: "tens", op: "filter", args: { items: "numbers", where: { field: "text", equals: "10" } },
    };
    const [line] = programToLines([step]);
    const s = tokens(line);
    expect(s).toContain("tens = filter(");
    expect(s).toContain("text");
    expect(s).toContain(" == ");
    expect(s).toContain('"10"');
  });

  it("renders answer with the question string", () => {
    const step: ProgramStep = {
      id: "result", op: "answer", args: { from: "ordered", question: "Does #10 score the first goal?" },
    };
    const [line] = programToLines([step]);
    const s = tokens(line);
    expect(s).toContain("result = answer(");
    expect(s).toContain("ordered");
    expect(s).toContain('"Does #10 score the first goal?"');
  });

  it("falls back to op(…) for an unknown op (default branch)", () => {
    const step: ProgramStep = { id: "x", op: "mystery_op", args: {} };
    const [line] = programToLines([step]);
    const s = tokens(line);
    expect(s).toBe("x = mystery_op(…)");
  });

  it("maps every program step to a line", () => {
    const program: ProgramStep[] = [
      { id: "frames", op: "sample_frames", args: { start_ms: 0, end_ms: 100, fps: 8 } },
      { id: "n", op: "count", args: { items: "frames" } },
    ];
    expect(programToLines(program)).toHaveLength(2);
  });
});

describe("commentFor", () => {
  // 2026-07-20 fix: comments used to be a static per-op map with baked-in narrative ("around the
  // goal moment", "keep only player #10") — wrong the moment the same op appeared in a different
  // program (a free-text "at the start of the video?" query rendered a "goal moment" comment on
  // production). commentFor derives the comment from THIS step's own args, so it can never claim
  // an intent the step doesn't actually have.

  it("sample_frames: singular for a one-frame window, plural with the real frame count otherwise", () => {
    // stride = round(1000/fps); a single-ms window at fps 1 samples exactly one frame.
    const single: ProgramStep = { id: "f", op: "sample_frames", args: { start_ms: 10000, end_ms: 10001, fps: 1 } };
    expect(commentFor(single)).toBe("# sample a single frame");

    // 3500-5000ms @ fps 8 -> stride 125 -> 13 frames (matches the real hero-clip capture).
    const many: ProgramStep = { id: "f", op: "sample_frames", args: { start_ms: 3500, end_ms: 5000, fps: 8 } };
    expect(commentFor(many)).toBe("# sample 13 frames across the window");
  });

  it("detect: names the real classes, and only mentions on_pitch when the arg is set", () => {
    const bare: ProgramStep = { id: "d", op: "detect", args: { frames: "f", classes: ["person"] } };
    expect(commentFor(bare)).toBe("# detect person in each frame");

    const onPitch: ProgramStep = { id: "d", op: "detect", args: { frames: "f", classes: ["person"], on_pitch: true } };
    expect(commentFor(onPitch)).toBe("# detect person in each frame, keeping only on-pitch boxes");
  });

  it("describe_scene, crop, read_text, count, answer: fixed mechanical descriptions", () => {
    expect(commentFor({ id: "s", op: "describe_scene", args: { frames: "f" } })).toBe("# caption each sampled frame");
    expect(commentFor({ id: "c", op: "crop", args: { detections: "d", region: "jersey" } }))
      .toBe('# crop the "jersey" region from each detection');
    expect(commentFor({ id: "t", op: "read_text", args: { crops: "c" } })).toBe("# read text out of each crop");
    expect(commentFor({ id: "n", op: "count", args: { items: "d" } })).toBe("# count the items");
    expect(commentFor({ id: "r", op: "answer", args: { from: "n", question: "anything at all?" } }))
      .toBe("# answer the question from the evidence");
  });

  it("filter and temporal_order: echo the real field/equals/by values, not a baked #10 or 'FIRST goal'", () => {
    const filterSeven: ProgramStep = {
      id: "sevens", op: "filter", args: { items: "numbers", where: { field: "text", equals: "7" } },
    };
    expect(commentFor(filterSeven)).toBe('# keep items where text == "7"');

    const order: ProgramStep = { id: "o", op: "temporal_order", args: { events: "goals", by: "timestamp" } };
    expect(commentFor(order)).toBe("# order events by timestamp");
  });

  it("falls back to `# <op>` for an unknown op", () => {
    expect(commentFor({ id: "x", op: "mystery_op", args: {} })).toBe("# mystery_op");
  });
});

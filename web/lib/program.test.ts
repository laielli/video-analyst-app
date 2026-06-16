import { describe, it, expect } from "vitest";
import { programToLines, COMMENTS } from "@/lib/program";
import type { ProgramStep } from "@/lib/types";

// program.ts is pure (it builds colored tokens from the DSL, no parsing). We assert one case per
// branch in lineFor: sample_frames, detect, describe_scene, filter, answer, and the default
// fallback — and that COMMENTS covers all 9 ops.

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
  });

  it("renders describe_scene over a frames binding", () => {
    const step: ProgramStep = { id: "scene", op: "describe_scene", args: { frames: "frames" } };
    const [line] = programToLines([step]);
    const s = tokens(line);
    expect(s).toBe("scene = describe_scene(frames)");
  });

  it("renders describe_scene with an optional max_captions", () => {
    const step: ProgramStep = { id: "scene", op: "describe_scene", args: { frames: "frames", max_captions: 3 } };
    const [line] = programToLines([step]);
    const s = tokens(line);
    expect(s).toContain("scene = describe_scene(frames");
    expect(s).toContain("max_captions=3");
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

describe("COMMENTS", () => {
  it("covers all 9 ops", () => {
    const ops = ["sample_frames", "detect", "describe_scene", "crop", "read_text", "filter", "count", "temporal_order", "answer"];
    for (const op of ops) {
      expect(COMMENTS[op]).toBeTruthy();
    }
    expect(Object.keys(COMMENTS).sort()).toEqual([...ops].sort());
  });
});

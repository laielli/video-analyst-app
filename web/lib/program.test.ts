import { describe, it, expect } from "vitest";
import { programToLines, COMMENTS } from "@/lib/program";
import type { ProgramStep } from "@/lib/types";

// Pure renderer: one ProgramStep -> a token line. We test one case per op branch in lineFor,
// asserting the joined token text so the coloring contract stays stable.

function text(step: ProgramStep): string {
  const [line] = programToLines([step]);
  return line.map((t) => t.t).join("");
}

describe("programToLines / lineFor", () => {
  it("renders sample_frames with start/end/fps", () => {
    const step: ProgramStep = {
      id: "frames",
      op: "sample_frames",
      args: { start_ms: 3500, end_ms: 5000, fps: 8 },
    };
    expect(text(step)).toBe("frames = sample_frames(3500, 5000, fps=8)");
  });

  it("renders detect with classes joined", () => {
    const step: ProgramStep = {
      id: "people",
      op: "detect",
      args: { frames: "frames", classes: ["person", "ball"] },
    };
    expect(text(step)).toBe('people = detect(frames, ["person", "ball"])');
  });

  it("renders crop with a quoted region", () => {
    const step: ProgramStep = {
      id: "jerseys",
      op: "crop",
      args: { detections: "people", region: "jersey" },
    };
    expect(text(step)).toBe('jerseys = crop(people, "jersey")');
  });

  it("renders read_text over crops", () => {
    const step: ProgramStep = { id: "numbers", op: "read_text", args: { crops: "jerseys" } };
    expect(text(step)).toBe("numbers = read_text(jerseys)");
  });

  it("renders filter with where.field == equals", () => {
    const step: ProgramStep = {
      id: "tens",
      op: "filter",
      args: { items: "numbers", where: { field: "text", equals: "10" } },
    };
    expect(text(step)).toBe('tens = filter(numbers, text == "10")');
  });

  it("renders count over items", () => {
    const step: ProgramStep = { id: "n", op: "count", args: { items: "people" } };
    expect(text(step)).toBe("n = count(people)");
  });

  it("renders temporal_order with by=", () => {
    const step: ProgramStep = {
      id: "ordered",
      op: "temporal_order",
      args: { events: "tens", by: "timestamp" },
    };
    expect(text(step)).toBe('ordered = temporal_order(tens, by="timestamp")');
  });

  it("renders answer with the question string", () => {
    const step: ProgramStep = {
      id: "result",
      op: "answer",
      args: { from: "ordered", question: "Does #10 score the first goal?" },
    };
    expect(text(step)).toBe('result = answer(ordered, "Does #10 score the first goal?")');
  });

  it("falls back to op(…) for an unknown op (default branch)", () => {
    const step: ProgramStep = { id: "x", op: "mystery_op", args: {} };
    expect(text(step)).toBe("x = mystery_op(…)");
  });

  it("maps an empty program to no lines", () => {
    expect(programToLines([])).toEqual([]);
  });
});

describe("COMMENTS", () => {
  it("covers all 8 ops", () => {
    expect(Object.keys(COMMENTS).sort()).toEqual(
      ["answer", "count", "crop", "detect", "filter", "read_text", "sample_frames", "temporal_order"].sort()
    );
  });
});

import { describe, it, expect } from "vitest";
import { programToLines, COMMENTS, type Tok } from "@/lib/program";
import type { ProgramStep } from "@/lib/types";

// program.ts is pure: programToLines(program) maps each step to colored tokens via lineFor's
// per-op switch. We test one case per branch (and the default fallback) and that COMMENTS covers
// all 8 ops.

const text = (toks: Tok[]) => toks.map((t) => t.t).join("");

describe("programToLines / lineFor", () => {
  it("sample_frames renders args + fps kwarg", () => {
    const step: ProgramStep = { id: "frames", op: "sample_frames", args: { start_ms: 3500, end_ms: 5000, fps: 8 } };
    const [line] = programToLines([step]);
    expect(text(line)).toBe("frames = sample_frames(3500, 5000, fps=8)");
  });

  it("detect joins classes", () => {
    const step: ProgramStep = { id: "people", op: "detect", args: { frames: "frames", classes: ["person", "ball"] } };
    const [line] = programToLines([step]);
    expect(text(line)).toBe('people = detect(frames, ["person", "ball"])');
  });

  it("crop renders region string", () => {
    const step: ProgramStep = { id: "jerseys", op: "crop", args: { detections: "people", region: "jersey" } };
    const [line] = programToLines([step]);
    expect(text(line)).toBe('jerseys = crop(people, "jersey")');
  });

  it("read_text renders the crops binding", () => {
    const step: ProgramStep = { id: "numbers", op: "read_text", args: { crops: "jerseys" } };
    const [line] = programToLines([step]);
    expect(text(line)).toBe("numbers = read_text(jerseys)");
  });

  it("filter renders where.field == equals", () => {
    const step: ProgramStep = { id: "tens", op: "filter", args: { items: "numbers", where: { field: "text", equals: "10" } } };
    const [line] = programToLines([step]);
    expect(text(line)).toBe('tens = filter(numbers, text == "10")');
  });

  it("count renders the items binding", () => {
    const step: ProgramStep = { id: "n", op: "count", args: { items: "people" } };
    const [line] = programToLines([step]);
    expect(text(line)).toBe("n = count(people)");
  });

  it("temporal_order renders by= kwarg", () => {
    const step: ProgramStep = { id: "ordered", op: "temporal_order", args: { events: "tens", by: "timestamp" } };
    const [line] = programToLines([step]);
    expect(text(line)).toBe('ordered = temporal_order(tens, by="timestamp")');
  });

  it("answer renders the question string", () => {
    const step: ProgramStep = { id: "result", op: "answer", args: { from: "ordered", question: "Does #10 score the first goal?" } };
    const [line] = programToLines([step]);
    expect(text(line)).toBe('result = answer(ordered, "Does #10 score the first goal?")');
  });

  it("default fallback for an unknown op", () => {
    const step: ProgramStep = { id: "mystery", op: "warp", args: {} };
    const [line] = programToLines([step]);
    expect(text(line)).toBe("mystery = warp(…)");
  });

  it("detect with no classes degrades to empty quotes", () => {
    const step: ProgramStep = { id: "people", op: "detect", args: { frames: "frames" } };
    const [line] = programToLines([step]);
    expect(text(line)).toBe('people = detect(frames, [""])');
  });

  it("maps a multi-step program in order", () => {
    const program: ProgramStep[] = [
      { id: "a", op: "count", args: { items: "x" } },
      { id: "b", op: "answer", args: { from: "a", question: "q" } },
    ];
    const lines = programToLines(program);
    expect(lines).toHaveLength(2);
    expect(text(lines[0])).toBe("a = count(x)");
    expect(text(lines[1])).toBe('b = answer(a, "q")');
  });
});

describe("COMMENTS", () => {
  it("covers all 8 ops", () => {
    expect(Object.keys(COMMENTS).sort()).toEqual(
      ["answer", "count", "crop", "detect", "filter", "read_text", "sample_frames", "temporal_order"]
    );
  });
});

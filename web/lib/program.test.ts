import { describe, it, expect } from "vitest";
import { programToLines, COMMENTS } from "./program";
import type { ProgramStep } from "./types";

// Render the token list of a single step to a plain string so assertions read the rendered line.
function render(step: ProgramStep): string {
  return programToLines([step])[0].map((t) => t.t).join("");
}

describe("programToLines / lineFor", () => {
  it("renders sample_frames with numeric args and fps keyword", () => {
    const line = render({ id: "frames", op: "sample_frames", args: { start_ms: 3500, end_ms: 5000, fps: 8 } });
    expect(line).toBe("frames = sample_frames(3500, 5000, fps=8)");
  });

  it("renders detect with classes joined", () => {
    const line = render({ id: "people", op: "detect", args: { frames: "frames", classes: ["person", "ball"] } });
    expect(line).toBe('people = detect(frames, ["person", "ball"])');
  });

  it("renders crop with region string", () => {
    const line = render({ id: "jerseys", op: "crop", args: { detections: "people", region: "jersey" } });
    expect(line).toBe('jerseys = crop(people, "jersey")');
  });

  it("renders read_text with one binding ref", () => {
    const line = render({ id: "numbers", op: "read_text", args: { crops: "jerseys" } });
    expect(line).toBe("numbers = read_text(jerseys)");
  });

  it("renders filter with where.field == equals", () => {
    const line = render({ id: "tens", op: "filter", args: { items: "numbers", where: { field: "text", equals: "10" } } });
    expect(line).toBe('tens = filter(numbers, text == "10")');
  });

  it("renders count", () => {
    const line = render({ id: "n", op: "count", args: { items: "people" } });
    expect(line).toBe("n = count(people)");
  });

  it("renders temporal_order with by keyword", () => {
    const line = render({ id: "ordered", op: "temporal_order", args: { events: "tens", by: "timestamp" } });
    expect(line).toBe('ordered = temporal_order(tens, by="timestamp")');
  });

  it("renders answer with the question string", () => {
    const line = render({ id: "result", op: "answer", args: { from: "ordered", question: "Does #10 score?" } });
    expect(line).toBe('result = answer(ordered, "Does #10 score?")');
  });

  it("falls back to a generic call for an unknown op (default branch)", () => {
    const line = render({ id: "x", op: "mystery", args: {} });
    expect(line).toBe("x = mystery(…)");
  });

  it("maps a whole program to one token row per step", () => {
    const program: ProgramStep[] = [
      { id: "frames", op: "sample_frames", args: { start_ms: 0, end_ms: 100, fps: 8 } },
      { id: "n", op: "count", args: { items: "frames" } },
    ];
    const rows = programToLines(program);
    expect(rows).toHaveLength(2);
  });
});

describe("COMMENTS", () => {
  it("has a comment for all 8 ops", () => {
    expect(Object.keys(COMMENTS).sort()).toEqual(
      ["answer", "count", "crop", "detect", "filter", "read_text", "sample_frames", "temporal_order"]
    );
    // each comment is a # line
    for (const v of Object.values(COMMENTS)) expect(v.startsWith("#")).toBe(true);
  });
});

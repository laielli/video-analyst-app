import { describe, it, expect } from "vitest";
import { programToLines, COMMENTS } from "./program";
import type { ProgramStep } from "./types";

// program.ts is pure: programToLines maps each step to coloring tokens; lineFor branches per op.
// We assert the rendered token TEXT (joined) per op branch rather than the token structure, so
// the tests stay readable but still bite on a real rendering change.

function text(step: ProgramStep): string {
  return programToLines([step])[0].map((t) => t.t).join("");
}

describe("lineFor — one case per op branch", () => {
  it("sample_frames renders start/end/fps", () => {
    const s: ProgramStep = {
      id: "frames", op: "sample_frames",
      args: { start_ms: 3500, end_ms: 5000, fps: 8 },
    };
    expect(text(s)).toBe("frames = sample_frames(3500, 5000, fps=8)");
  });

  it("detect joins classes into a bracketed string list", () => {
    const s: ProgramStep = {
      id: "people", op: "detect",
      args: { frames: "frames", classes: ["person", "ball"] },
    };
    expect(text(s)).toBe('people = detect(frames, ["person", "ball"])');
  });

  it("crop renders detections + region", () => {
    const s: ProgramStep = {
      id: "jerseys", op: "crop",
      args: { detections: "people", region: "jersey" },
    };
    expect(text(s)).toBe('jerseys = crop(people, "jersey")');
  });

  it("read_text renders crops", () => {
    const s: ProgramStep = { id: "numbers", op: "read_text", args: { crops: "jerseys" } };
    expect(text(s)).toBe("numbers = read_text(jerseys)");
  });

  it("filter renders where.field == equals", () => {
    const s: ProgramStep = {
      id: "tens", op: "filter",
      args: { items: "numbers", where: { field: "text", equals: "10" } },
    };
    expect(text(s)).toBe('tens = filter(numbers, text == "10")');
  });

  it("count renders items", () => {
    const s: ProgramStep = { id: "n", op: "count", args: { items: "people" } };
    expect(text(s)).toBe("n = count(people)");
  });

  it("temporal_order renders events + by", () => {
    const s: ProgramStep = {
      id: "ordered", op: "temporal_order",
      args: { events: "tens", by: "timestamp" },
    };
    expect(text(s)).toBe('ordered = temporal_order(tens, by="timestamp")');
  });

  it("answer renders the from binding + question string", () => {
    const s: ProgramStep = {
      id: "result", op: "answer",
      args: { from: "ordered", question: "Does #10 score the first goal?" },
    };
    expect(text(s)).toBe('result = answer(ordered, "Does #10 score the first goal?")');
  });

  it("default fallback renders op(…) for an unknown op", () => {
    const s: ProgramStep = { id: "weird", op: "mystery_op", args: {} };
    expect(text(s)).toBe("weird = mystery_op(…)");
  });
});

describe("programToLines", () => {
  it("maps each program step to a token line", () => {
    const program: ProgramStep[] = [
      { id: "frames", op: "sample_frames", args: { start_ms: 0, end_ms: 100, fps: 8 } },
      { id: "n", op: "count", args: { items: "frames" } },
    ];
    const lines = programToLines(program);
    expect(lines).toHaveLength(2);
    expect(lines[0].some((t) => t.c === "fn" && t.t === "sample_frames")).toBe(true);
  });
});

describe("COMMENTS", () => {
  it("covers all 8 ops", () => {
    expect(Object.keys(COMMENTS).sort()).toEqual(
      ["answer", "count", "crop", "detect", "filter", "read_text", "sample_frames", "temporal_order"],
    );
  });
});

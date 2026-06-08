import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import EvidencePanel from "@/components/EvidencePanel";
import type { StepResult } from "@/lib/types";

// EvidencePanel maps a step's evidence ts to a still (frameSrc): ts 0 -> empty state; ts <= 4200
// -> goal still; ts > 4200 -> scorer still. Overlays render with a tone class + label, and the
// program provenance tag renders when programSource is set.

function step(partial: Partial<StepResult>): StepResult {
  return { id: "s", op: "detect", status: "done", source: "cached", ...partial };
}

describe("EvidencePanel", () => {
  it("ts 0 (no evidence) shows the empty state, no img", () => {
    render(<EvidencePanel step={step({ evidence: { frame_ts_ms: 0, overlays: [] } })} />);
    expect(screen.getByText(/No evidence frame for this step/i)).toBeInTheDocument();
    expect(document.querySelector("img")).toBeNull();
  });

  it("ts <= 4200 shows the goal still", () => {
    render(<EvidencePanel step={step({ evidence: { frame_ts_ms: 4000, overlays: [] } })} />);
    const img = document.querySelector("img") as HTMLImageElement;
    expect(img).toBeTruthy();
    expect(img.getAttribute("src")).toBe("/frames/goal-4000.jpg");
  });

  it("ts > 4200 shows the scorer still", () => {
    render(<EvidencePanel step={step({ evidence: { frame_ts_ms: 4625, overlays: [] } })} />);
    const img = document.querySelector("img") as HTMLImageElement;
    expect(img.getAttribute("src")).toBe("/frames/scorer-4625.jpg");
  });

  it("renders overlays with tone class + label", () => {
    const s = step({
      evidence: {
        frame_ts_ms: 4625,
        overlays: [{ box: { x: 0.1, y: 0.2, w: 0.3, h: 0.4 }, label: "#10 scored", tone: "green", kind: "box" }],
      },
    });
    render(<EvidencePanel step={s} />);
    const ov = document.querySelector(".ov.green");
    expect(ov).toBeInTheDocument();
    expect(screen.getByText("#10 scored")).toBeInTheDocument();
  });

  it("renders the program provenance tag", () => {
    render(<EvidencePanel step={step({ evidence: { frame_ts_ms: 4625, overlays: [] } })} programSource="ungrounded" />);
    const tag = document.querySelector(".prov-tag.ungrounded");
    expect(tag).toBeInTheDocument();
    expect(tag?.textContent).toBe("ungrounded");
  });

  it("null step degrades to the empty state", () => {
    render(<EvidencePanel step={null} />);
    expect(screen.getByText(/No evidence frame for this step/i)).toBeInTheDocument();
  });
});

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

  it("renders overlays with tone class + label + box coords", () => {
    const s = step({
      evidence: {
        frame_ts_ms: 4625,
        // distinctive non-round coords so a hard-coded value couldn't coincidentally match.
        overlays: [{ box: { x: 0.111, y: 0.222, w: 0.333, h: 0.444 }, label: "#10 scored", tone: "green", kind: "box" }],
      },
    });
    render(<EvidencePanel step={s} />);
    const ov = document.querySelector(".ov.green") as HTMLElement;
    expect(ov).toBeInTheDocument();
    expect(screen.getByText("#10 scored")).toBeInTheDocument();
    // box positioned from its normalized coords (left=x*100% ...). parseFloat is agnostic to
    // CSSOM trailing-zero formatting; deleting/hard-coding the coord logic would fail these.
    expect(parseFloat(ov.style.left)).toBeCloseTo(11.1, 5);
    expect(parseFloat(ov.style.top)).toBeCloseTo(22.2, 5);
    expect(parseFloat(ov.style.width)).toBeCloseTo(33.3, 5);
    expect(parseFloat(ov.style.height)).toBeCloseTo(44.4, 5);
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

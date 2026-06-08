import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import EvidencePanel from "@/components/EvidencePanel";
import type { StepResult } from "@/lib/types";

// EvidencePanel maps a step's evidence ts -> a frame still (or an empty state at ts 0), renders
// overlays with their tone class + label, and shows the program-source provenance tag.

function step(over: Partial<StepResult> = {}): StepResult {
  return {
    id: "s", op: "detect", producer: "Azure AI Vision · detect", status: "done", source: "cached",
    input_label: "5 frames", output_label: "5 people", confidence: 0.75,
    evidence: { frame_ts_ms: 4000, overlays: [] }, ...over,
  };
}

describe("EvidencePanel", () => {
  it("ts 0 -> 'No evidence frame' empty state (no img)", () => {
    const s = step({ evidence: { frame_ts_ms: 0, overlays: [] } });
    const { container } = render(<EvidencePanel step={s} />);
    expect(screen.getByLabelText("No evidence frame for this step")).toBeInTheDocument();
    expect(container.querySelector("img")).toBeNull();
  });

  it("ts <= 4200 -> the goal still", () => {
    const s = step({ evidence: { frame_ts_ms: 4000, overlays: [] } });
    const { container } = render(<EvidencePanel step={s} />);
    expect(container.querySelector("img")!.getAttribute("src")).toBe("/frames/goal-4000.jpg");
  });

  it("ts > 4200 -> the scorer still", () => {
    const s = step({ evidence: { frame_ts_ms: 4625, overlays: [] } });
    const { container } = render(<EvidencePanel step={s} />);
    expect(container.querySelector("img")!.getAttribute("src")).toBe("/frames/scorer-4625.jpg");
  });

  it("overlays render with tone class + label", () => {
    const s = step({
      evidence: {
        frame_ts_ms: 4625,
        overlays: [{ box: { x: 0.1, y: 0.2, w: 0.3, h: 0.4 }, label: "person 0.75", tone: "azure", confidence: 0.75 }],
      },
    });
    const { container } = render(<EvidencePanel step={s} />);
    const ov = container.querySelector(".ov.azure");
    expect(ov).toBeTruthy();
    expect(screen.getByText("person 0.75")).toBeInTheDocument();
  });

  it("programSource provenance tag renders", () => {
    const { container } = render(<EvidencePanel step={step()} programSource="ungrounded" />);
    const tag = container.querySelector(".prov-tag.ungrounded");
    expect(tag).toBeTruthy();
    expect(tag!.textContent).toBe("ungrounded");
  });

  it("renders the step note when present", () => {
    const s = step({ note: "no detections in the sampled frames" });
    render(<EvidencePanel step={s} />);
    expect(screen.getByText("no detections in the sampled frames")).toBeInTheDocument();
  });
});

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import EvidencePanel from "./EvidencePanel";
import type { StepResult, Overlay } from "@/lib/types";

function step(over: Partial<StepResult> = {}): StepResult {
  return {
    id: "s", op: "detect", producer: "Azure AI Vision · detect", status: "done",
    source: "cached", output_label: "5 people", input_label: "12 frames",
    evidence: { frame_ts_ms: 4625, overlays: [] }, ...over,
  };
}
const overlay = (o: Partial<Overlay> = {}): Overlay => ({
  box: { x: 0.1, y: 0.2, w: 0.3, h: 0.4 }, tone: "azure", label: "person 0.75", ...o,
});

describe("EvidencePanel", () => {
  it("ts === 0 -> empty state, no frame", () => {
    render(<EvidencePanel step={step({ evidence: { frame_ts_ms: 0, overlays: [] } })} />);
    expect(screen.getByText(/No evidence frame for this step/i)).toBeInTheDocument();
    expect(screen.queryByRole("img")).toBeNull();
    expect(screen.getByText("no frame")).toBeInTheDocument();
  });

  it("ts <= 4200 -> the goal still", () => {
    render(<EvidencePanel step={step({ evidence: { frame_ts_ms: 4000, overlays: [] } })} />);
    expect(screen.getByRole("img").getAttribute("src")).toBe("/frames/goal-4000.jpg");
  });

  it("ts > 4200 -> the scorer still", () => {
    render(<EvidencePanel step={step({ evidence: { frame_ts_ms: 4625, overlays: [] } })} />);
    expect(screen.getByRole("img").getAttribute("src")).toBe("/frames/scorer-4625.jpg");
  });

  it("overlays render with tone class + label + confidence", () => {
    const { container } = render(
      <EvidencePanel step={step({ evidence: { frame_ts_ms: 4625, overlays: [overlay({ tone: "green", label: "goal", confidence: 0.88 })] } })} />
    );
    const ov = container.querySelector(".ov.green");
    expect(ov).toBeTruthy();
    expect(screen.getByText("goal")).toBeInTheDocument();
    expect(screen.getByText(/0\.88/)).toBeInTheDocument();
  });

  it("renders the program-source provenance tag when provided", () => {
    const { container } = render(<EvidencePanel step={step()} programSource="ungrounded" />);
    const tag = container.querySelector(".prov-tag.ungrounded");
    expect(tag).toBeTruthy();
    expect(tag?.textContent).toBe("ungrounded");
  });

  it("renders the step note when present", () => {
    render(<EvidencePanel step={step({ note: "no detections in the sampled frames" })} />);
    expect(screen.getByText("no detections in the sampled frames")).toBeInTheDocument();
  });
});

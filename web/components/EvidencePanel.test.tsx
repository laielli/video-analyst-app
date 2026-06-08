import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import EvidencePanel from "./EvidencePanel";
import type { StepResult } from "@/lib/types";

// frameSrc(ts): ts === 0 -> no frame (empty state); ts <= 4200 -> goal still; ts > 4200 -> scorer still.

function step(over: Partial<StepResult> = {}): StepResult {
  return {
    id: "s", op: "detect", status: "done", source: "cached",
    input_label: "1 frame", output_label: "1 people", confidence: 0.75,
    evidence: { frame_ts_ms: 0, overlays: [] },
    ...over,
  };
}

function img() {
  return document.querySelector(".frame img") as HTMLImageElement | null;
}

describe("EvidencePanel — frameSrc mapping", () => {
  it("ts 0 -> No evidence frame empty state, no img", () => {
    render(<EvidencePanel step={step({ evidence: { frame_ts_ms: 0, overlays: [] } })} />);
    expect(screen.getByLabelText("No evidence frame for this step")).toBeInTheDocument();
    expect(img()).toBeNull();
  });

  it("ts <= 4200 -> goal still", () => {
    render(<EvidencePanel step={step({ evidence: { frame_ts_ms: 4000, overlays: [] } })} />);
    expect(img()?.getAttribute("src")).toBe("/frames/goal-4000.jpg");
  });

  it("ts > 4200 -> scorer still", () => {
    render(<EvidencePanel step={step({ evidence: { frame_ts_ms: 4625, overlays: [] } })} />);
    expect(img()?.getAttribute("src")).toBe("/frames/scorer-4625.jpg");
  });
});

describe("EvidencePanel — overlays", () => {
  it("renders overlays with tone class + label + confidence", () => {
    const s = step({
      evidence: {
        frame_ts_ms: 4625,
        overlays: [
          { box: { x: 0.1, y: 0.2, w: 0.3, h: 0.4 }, label: "#10 scored", tone: "green", confidence: 0.75 },
        ],
      },
    });
    render(<EvidencePanel step={s} />);
    const ov = document.querySelector(".ov.green");
    expect(ov).toBeInTheDocument();
    expect(screen.getByText("#10 scored")).toBeInTheDocument();
    // the overlay's own confidence badge (scoped to the overlay's .conf span).
    expect(ov?.querySelector(".ov-tag .conf")?.textContent).toContain("0.75");
  });
});

describe("EvidencePanel — provenance tag", () => {
  it("renders the programSource prov-tag", () => {
    render(<EvidencePanel step={step({ evidence: { frame_ts_ms: 4000, overlays: [] } })} programSource="ungrounded" />);
    const tag = document.querySelector(".prov-tag.ungrounded");
    expect(tag).toBeInTheDocument();
    expect(tag?.textContent).toBe("ungrounded");
  });

  it("omits the prov-tag when programSource is undefined", () => {
    render(<EvidencePanel step={step()} />);
    expect(document.querySelector(".prov-tag")).toBeNull();
  });
});

describe("EvidencePanel — readout", () => {
  it("shows producer/input/output and the step note", () => {
    const s = step({ producer: "Azure AI Vision · detect", note: "no detections in the sampled frames" });
    render(<EvidencePanel step={s} />);
    expect(screen.getByText("1 people")).toBeInTheDocument();
    expect(screen.getByText("no detections in the sampled frames")).toBeInTheDocument();
  });
});

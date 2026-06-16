import type { StepResult, ProgramSource } from "@/lib/types";

const pct = (n: number) => `${(n * 100).toFixed(3)}%`;
const ms = (n: number) => {
  const s = Math.floor(n / 1000), cs = Math.round((n % 1000) / 10);
  return `00:00:${String(s).padStart(2, "0")}:${String(cs).padStart(2, "0")}`;
};

// frameSrc stays clip-scoped to the single v1 clip's hero stills. A free-text run over the same
// clip uses these same stills. When a step carries no evidence frame (ts === 0 — e.g. an
// ungrounded answer step), there is no meaningful frame to show, so we degrade to no still rather
// than show a misleading default frame.
function frameSrc(ts: number): string | null {
  if (!ts) return null; // no/zero evidence ts -> no frame (graceful, not a misleading still)
  return ts <= 4200 ? "/frames/goal-4000.jpg" : "/frames/scorer-4625.jpg";
}

export default function EvidencePanel({
  step, programSource,
}: { step: StepResult | null; programSource?: ProgramSource }) {
  const ev = step?.evidence;
  const ts = ev?.frame_ts_ms ?? 0;
  const conf = step?.confidence;
  const src = frameSrc(ts);

  return (
    <section className="panel evidence" aria-label="Evidence — inspectable">
      <div className="panel-head">
        <div className="panel-title">Evidence <span className="sub">(inspectable)</span></div>
        <div className="panel-meta">
          {/* Program provenance: live compile vs. the honest ungrounded state (vs. pinned replay). */}
          {programSource && <span className={`prov-tag ${programSource}`}>{programSource}</span>}
          <span style={{ marginLeft: programSource ? 8 : 0 }}>{ts ? `frame ${ms(ts)}` : "no frame"}</span>
        </div>
      </div>
      <div className="evi-body">
        <div className="frame">
          {src ? (
            <>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={src} alt="analyzed video frame" />
              <div className="tc">{ms(ts)}</div>
            </>
          ) : (
            <div className="frame-empty" aria-label="No evidence frame for this step">
              No evidence frame for this step.
            </div>
          )}
          {(ev?.overlays ?? []).map((o, i) => (
            <div
              key={i}
              className={`ov ${o.tone}`}
              style={{ left: pct(o.box.x), top: pct(o.box.y), width: pct(o.box.w), height: pct(o.box.h) }}
            >
              {o.label && (
                <span className="ov-tag">
                  {o.label}
                  {o.confidence != null && <span className="conf"> {o.confidence.toFixed(2)}</span>}
                </span>
              )}
            </div>
          ))}
        </div>

        <p className="evi-cap">
          Every result is <b>traceable to a frame</b> — the box above is what the active step
          {step ? <> (<code>{step.producer ?? step.op}</code>)</> : ""} produced.
        </p>

        <div className="readout">
          <div className="row"><div className="k">Producer step</div><div className="v">{step?.producer ?? step?.op ?? "—"}</div></div>
          <div className="row"><div className="k">Input</div><div className="v">{step?.input_label ?? "—"}</div></div>
          <div className="row">
            <div className="k">Output</div>
            <div className="v">
              <span className={`pill ${step?.source === "pinned" ? "pinned" : ""}`}>{step?.output_label ?? "—"}</span>
              {step?.source && <span className={`src-tag ${step.source}`}>{step.source}</span>}
            </div>
          </div>
          {step?.op === "describe_scene" && (
            // Scene caption gets its own readout row: describe_scene's whole-image caption is free
            // text (the answer IS the caption), so surface it in full rather than only as the
            // truncated Output pill. output_label carries the caption (or "(no scene caption)").
            <div className="row">
              <div className="k">Scene caption</div>
              <div className="v scene-caption">{step.output_label ?? "—"}</div>
            </div>
          )}
          <div className="row">
            <div className="k">Confidence</div>
            <div className="v">
              <div className="conf-bar"><div className="conf-fill" style={{ width: conf != null ? pct(conf) : "0%" }} /></div>
              <span>{conf != null ? conf.toFixed(2) : "—"}</span>
            </div>
          </div>
        </div>
        {step?.note && <div className="note">{step.note}</div>}
      </div>
    </section>
  );
}

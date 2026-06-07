import type { StepResult, Meta } from "@/lib/types";

const pct = (n: number) => `${(n * 100).toFixed(3)}%`;
const ms = (n: number) => {
  const s = Math.floor(n / 1000), cs = Math.round((n % 1000) / 10);
  return `00:00:${String(s).padStart(2, "0")}:${String(cs).padStart(2, "0")}`;
};

// Clip-scoped hero stills for the single v1 clip. A step with no evidence frame (ts <= 0)
// returns null so we degrade gracefully instead of showing a misleading frame.
function frameSrc(ts: number): string | null {
  if (!ts || ts <= 0) return null;
  return ts <= 4200 ? "/frames/goal-4000.jpg" : "/frames/scorer-4625.jpg";
}

export default function EvidencePanel({
  step, programSource,
}: { step: StepResult | null; programSource?: Meta["program_source"] }) {
  const ev = step?.evidence;
  const rawTs = ev?.frame_ts_ms ?? 0;
  const src = frameSrc(rawTs);
  const ts = rawTs || 4625;              // label fallback only; src may still be null
  const conf = step?.confidence;
  const overlays = src ? (ev?.overlays ?? []) : []; // overlays are meaningless without a frame

  return (
    <section className="panel evidence" aria-label="Evidence — inspectable">
      <div className="panel-head">
        <div className="panel-title">Evidence <span className="sub">(inspectable)</span></div>
        <div className="panel-meta">
          {programSource && <span className={`src-tag ${programSource}`} title="Program provenance">{programSource}</span>}
          <span>frame {ms(ts)}</span>
        </div>
      </div>
      <div className="evi-body">
        <div className="frame">
          {src ? (
            <>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={src} alt="analyzed video frame" />
              <div className="tc">{ms(ts)}</div>
              {overlays.map((o, i) => (
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
            </>
          ) : (
            <div className="frame-empty">No evidence frame for this step.</div>
          )}
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

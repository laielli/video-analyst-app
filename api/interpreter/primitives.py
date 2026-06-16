"""
Op implementations. Each is fn(step, env, cache) -> (binding, step_result):
  - binding: the value stored under step["id"] for later steps to reference.
  - step_result: the run-doc trace entry (status, provenance, EVIDENCE overlays).

Replay mode: detect/read_text read from the cache (real Azure outputs + pins);
crop/filter/count/temporal_order/answer are local compute. No live calls.

Binding kinds and item shapes:
  frames      items=[{frame_ts_ms}]
  detections  items=[{det_id,cls,confidence,box,frame_ts_ms}]
  captions    items=[{text,confidence,box,frame_ts_ms}]  (scene caption per frame; box full-frame)
  crops       items=[{crop_id,det_id,box,person_box,person_conf,frame_ts_ms}]
  texts       items=[{det_id,text,confidence,source,note?,crop_box,person_box,frame_ts_ms}]
  number      value=int
  ordered     value={events,first,subject_dets,subject_label,subject_is_first_scorer}
  answer      value={answer,verdict,question,yes,grounded,reason?}
"""
from __future__ import annotations

from limits import (
    MAX_CAPTION_LEN,
    MAX_SAMPLED_FRAMES,
    MAX_SCENE_CAPTIONS,
    ProgramLimitExceeded,
    sampled_frame_count,
)


def _evi(frame_ts_ms, overlays):
    return {"frame_ts_ms": int(frame_ts_ms), "overlays": overlays}


def op_sample_frames(step, env, cache):
    a = step["args"]
    start, end, fps = a["start_ms"], a["end_ms"], a["fps"]
    stride = max(1, round(1000 / fps))
    # Load-bearing runtime guard (trust boundary): bound the would-be length of
    # range(start, end+1, stride) BEFORE materializing it. A post-hoc len() on the built list
    # fires only after a 60M-frame bomb has already OOM'd the process, so the cap MUST sit here.
    # The count uses the same stride math (shared helper) so the validator and this guard agree.
    would_build = sampled_frame_count(start, end, fps)
    if would_build > MAX_SAMPLED_FRAMES:
        raise ProgramLimitExceeded(
            f"sample_frames would materialize {would_build} frames; max is {MAX_SAMPLED_FRAMES}"
        )
    ts = list(range(start, end + 1, stride))
    binding = {"kind": "frames", "items": [{"frame_ts_ms": t} for t in ts]}
    result = {
        "id": step["id"], "op": "sample_frames", "producer": "sample_frames",
        "status": "done", "source": "live", "inputs": [],
        "input_label": f"clip {start}-{end}ms",
        "output_label": f"{len(ts)} frames @ {fps}fps", "confidence": None,
        "evidence": _evi(ts[len(ts) // 2] if ts else start, []),
    }
    return binding, result


def op_detect(step, env, cache):
    a = step["args"]
    frames = env[a["frames"]]["items"]
    classes = [c.lower() for c in a["classes"]]
    dets = []
    for f in frames:
        for d in cache.detections_at(f["frame_ts_ms"]):
            if d["cls"].lower() in classes:
                dets.append({**d, "frame_ts_ms": f["frame_ts_ms"]})
    focus_ts = dets[-1]["frame_ts_ms"] if dets else (frames[-1]["frame_ts_ms"] if frames else 0)
    overlays = [
        {"box": d["box"], "label": f"{d['cls']} {d['confidence']:.2f}",
         "confidence": d["confidence"], "tone": "azure", "kind": "box"}
        for d in dets if d["frame_ts_ms"] == focus_ts
    ]
    noun = "people" if classes == ["person"] else "objects"
    result = {
        "id": step["id"], "op": "detect", "producer": "Azure AI Vision · detect",
        "status": "done" if dets else "empty", "source": "cached",
        "inputs": [a["frames"]], "input_label": f"{len(frames)} frames",
        "output_label": f"{len(dets)} {noun}",
        "confidence": max((d["confidence"] for d in dets), default=None),
        "evidence": _evi(focus_ts, overlays),
    }
    if not dets:
        result["note"] = "no detections in the sampled frames"
    return {"kind": "detections", "items": dets}, result


_FULL_FRAME_BOX = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}


def op_describe_scene(step, env, cache):
    """Scene captioning (Azure Image Analysis CAPTION). Replay mode: read one whole-image caption
    per sampled frame from the cache's `captions` slice (mirrors op_detect's per-ts read). Each
    caption's model-generated free text is TRUNCATED to MAX_CAPTION_LEN at the boundary (security:
    captions enter a binding/answer/run-doc with no schema maxLength). Output kind: captions."""
    a = step["args"]
    frames = env[a["frames"]]["items"]
    cap = a.get("max_captions", MAX_SCENE_CAPTIONS)
    items = []
    for f in frames:
        ts = f["frame_ts_ms"]
        for c in cache.captions_at(ts)[:1]:  # CAPTION returns ONE whole-image caption per frame
            text = (c.get("text") or "")[:MAX_CAPTION_LEN]  # bound model free-text at the boundary
            box = c.get("box") or _FULL_FRAME_BOX
            items.append({"text": text, "confidence": c.get("confidence"),
                          "box": box, "frame_ts_ms": ts})
    items = items[:cap]
    focus_ts = items[0]["frame_ts_ms"] if items else (frames[0]["frame_ts_ms"] if frames else 0)
    overlays = [
        {"box": c["box"], "label": f"scene: {c['text']}", "confidence": c.get("confidence"),
         "tone": "teal", "kind": "crop"}
        for c in items if c["frame_ts_ms"] == focus_ts
    ]
    result = {
        "id": step["id"], "op": "describe_scene", "producer": "Azure AI Vision · describe_scene",
        "status": "done" if items else "empty", "source": "cached",
        "inputs": [a["frames"]], "input_label": f"{len(frames)} frames",
        "output_label": (items[0]["text"] if items else "(no scene caption)"),
        "confidence": next((c["confidence"] for c in items if c["confidence"] is not None), None),
        "evidence": _evi(focus_ts, overlays),
    }
    if not items:
        # D-DR5 diagnostic-empty: WHAT was searched + WHY, never a bare "No evidence".
        result["note"] = (f"no scene captions cached for the {len(frames)} sampled frames "
                          "(the window has no precomputed CAPTION analysis)")
    return {"kind": "captions", "items": items}, result


def op_crop(step, env, cache):
    a = step["args"]
    dets = env[a["detections"]]["items"]
    region = a["region"]
    crops = []
    for i, d in enumerate(dets):
        b = d["box"]
        if region == "jersey":
            box = {"x": round(b["x"] + 0.22 * b["w"], 4), "y": round(b["y"] + 0.10 * b["h"], 4),
                   "w": round(0.56 * b["w"], 4), "h": round(0.22 * b["h"], 4)}
        else:
            box = dict(b)
        crops.append({"crop_id": f"c{i}", "det_id": d["det_id"], "box": box,
                      "person_box": b, "person_conf": d.get("confidence"), "frame_ts_ms": d["frame_ts_ms"]})
    focus_ts = crops[-1]["frame_ts_ms"] if crops else 0
    overlays = [{"box": c["box"], "label": "crop_jersey" if region == "jersey" else f"crop_{region}",
                 "tone": "teal", "kind": "crop"} for c in crops if c["frame_ts_ms"] == focus_ts]
    result = {
        "id": step["id"], "op": "crop", "producer": f"crop_{region}",
        "status": "done" if crops else "empty", "source": "live",
        "inputs": [a["detections"]], "input_label": f"{len(dets)} boxes",
        "output_label": f"{len(crops)} {region} crops", "confidence": None,
        "evidence": _evi(focus_ts, overlays),
    }
    return {"kind": "crops", "items": crops}, result


def op_read_text(step, env, cache):
    a = step["args"]
    crops = env[a["crops"]]["items"]
    texts, sources, notes = [], set(), []
    for c in crops:
        rt = cache.read_text_for(c["det_id"])
        if rt and rt.get("text"):
            texts.append({"det_id": c["det_id"], "text": rt["text"], "confidence": rt.get("confidence"),
                          "source": rt.get("source", "live"), "note": rt.get("note"),
                          "crop_box": c["box"], "person_box": c["person_box"], "frame_ts_ms": c["frame_ts_ms"]})
            sources.add(rt.get("source", "live"))
            if rt.get("note"):
                notes.append(rt["note"])
    step_source = "pinned" if "pinned" in sources else ("cached" if "cached" in sources else "live")
    focus_ts = texts[-1]["frame_ts_ms"] if texts else (crops[-1]["frame_ts_ms"] if crops else 0)
    overlays = [{"box": t["crop_box"], "label": f"read_text -> {t['text']}",
                 "confidence": t.get("confidence"), "tone": "teal", "kind": "crop"}
                for t in texts if t["frame_ts_ms"] == focus_ts]
    result = {
        "id": step["id"], "op": "read_text", "producer": "read_text",
        "status": "done" if texts else "empty", "source": step_source,
        "inputs": [a["crops"]], "input_label": f"{len(crops)} crops",
        "output_label": ", ".join(sorted({t["text"] for t in texts})) or "(none legible)",
        "confidence": next((t["confidence"] for t in texts if t["confidence"] is not None), None),
        "evidence": _evi(focus_ts, overlays),
    }
    if notes:
        result["note"] = notes[0]
    elif not texts:
        result["note"] = "no legible text in the jersey crops (read returned nothing)"
    return {"kind": "texts", "items": texts}, result


def op_filter(step, env, cache):
    a = step["args"]
    src = env[a["items"]]
    field, val = a["where"]["field"], a["where"]["equals"]
    kept = [it for it in src["items"] if str(it.get(field)) == str(val)]
    overlays, focus_ts = [], 0
    for it in kept:
        pb = it.get("person_box") or it.get("box")
        if pb:
            overlays.append({"box": pb, "label": f"#{val}", "tone": "teal", "kind": "box"})
            focus_ts = it.get("frame_ts_ms", focus_ts)
    result = {
        "id": step["id"], "op": "filter", "producer": "filter",
        "status": "done" if kept else "empty", "source": "live",
        "inputs": [a["items"]], "input_label": f"{src['kind']} where {field} == {val}",
        "output_label": f"{len(kept)} match" + (f" · #{val}" if kept else ""), "confidence": None,
        "evidence": _evi(focus_ts, overlays),
    }
    if not kept:
        result["note"] = f"no items with {field} == {val}"
    return {"kind": src["kind"], "items": kept}, result


def op_count(step, env, cache):
    a = step["args"]
    src = env[a["items"]]
    n = len(src["items"])
    result = {
        "id": step["id"], "op": "count", "producer": "count", "status": "done", "source": "live",
        "inputs": [a["items"]], "input_label": src["kind"], "output_label": str(n),
        "confidence": None, "evidence": _evi(0, []),
    }
    return {"kind": "number", "value": n}, result


def _subject_label(items) -> str | None:
    """Derive the '#<n>' subject label from a filtered/ordered collection's matched text.
    The hero path filters jersey OCR on `text == "10"`, so the matched item's `text` is the
    subject number. Returns None when there is no legible subject so callers never fabricate
    a '#10' (the de-hardcoded subject framing, plan Phase 0)."""
    for it in items or []:
        val = it.get("text")
        if val not in (None, ""):
            return f"#{val}"
    return None


def op_temporal_order(step, env, cache):
    a = step["args"]
    subj = env[a["events"]]
    subject_items = subj.get("items", [])
    subject_dets = {it.get("det_id") for it in subject_items if it.get("det_id")}
    subject_label = _subject_label(subject_items)
    goals = cache.goal_events()
    first = goals[0] if goals else None
    subject_is_first = bool(first and first.get("scorer_det") in subject_dets)
    overlays, focus_ts = [], 0
    if first:
        focus_ts = first["ts_ms"]
        if first.get("box"):
            overlays.append({"box": first["box"], "label": f"goal {first['ts_ms']}ms", "tone": "green", "kind": "goal"})
    result = {
        "id": step["id"], "op": "temporal_order", "producer": "temporal_order",
        "status": "done" if first else "empty", "source": "live",
        "inputs": [a["events"]], "input_label": f"{len(subject_dets)} subject(s) + {len(goals)} goal(s)",
        "output_label": (f"goal @ {first['ts_ms']}ms (first)" if first else "no goal events"),
        "confidence": None, "evidence": _evi(focus_ts, overlays),
    }
    value = {"events": goals, "first": first, "subject_dets": sorted(subject_dets),
             "subject_label": subject_label,  # derived '#<n>' (None if unknown) — never a literal
             "subject_is_first_scorer": subject_is_first}
    return {"kind": "ordered", "value": value}, result


# ---- generalized answer synthesis (plan Phase 0) ---------------------------------------
# op_answer dispatches on the KIND of its `from` binding and synthesizes a deterministic
# answer/verdict (replay-mode, no live call). When it is handed a binding it cannot ground
# (empty temporal subject, count == 0, unsupported kind) it emits the honest ungrounded
# discriminator (grounded=False + reason) instead of a fabricated verdict (Q1 -> A).

def _answer_temporal(src, q):
    """boolean/temporal: 'did <subject> score first?'. Subject is derived, never a literal #10."""
    v = src["value"]
    subj = v.get("subject_label")
    if not v.get("first"):
        return None, None, None, "no-grounded-answer"   # no goal events -> ungrounded
    if not subj:
        # A first goal exists, but the subject filter matched nobody, so we never located the
        # asked subject. We cannot honestly say they did or didn't score it — a confident "No"
        # would overclaim about a player we couldn't even find. Ground out instead (Q1 -> A).
        return None, None, None, "no-grounded-answer"
    yes = bool(v.get("subject_is_first_scorer"))
    if yes:
        verdict = f"Yes — {subj} scored the first goal"
    else:
        verdict = f"No — the first goal was not scored by {subj}"
    return ("Yes" if yes else "No"), verdict, yes, None


def _answer_count(src, q):
    """count: 'how many ...?' -> 'N'."""
    n = src.get("value", 0)
    if n <= 0:
        return None, None, None, "no-grounded-answer"   # nothing to count -> ungrounded
    return str(n), f"{n} found", None, None


def _answer_text(src, q):
    """read_text: 'what number ...?' -> the legible read value(s)."""
    items = src.get("items", [])
    texts = [it["text"] for it in items if it.get("text") not in (None, "")]
    if not texts:
        return None, None, None, "no-grounded-answer"   # nothing legible -> ungrounded
    uniq = sorted(set(texts))
    readout = ", ".join(uniq)
    return readout, f"Read: {readout}", None, None


def _answer_presence(src, q):
    """existence: 'is there a ...?' over detections/crops/frames -> Yes/No by non-emptiness."""
    items = src.get("items", [])
    present = len(items) > 0
    if not present:
        return None, None, None, "no-grounded-answer"   # empty -> honest ungrounded, not a fake "No"
    return "Yes", f"Yes — found {len(items)}", True, None


def _answer_caption(src, q):
    """scene: 'what is happening?' over captions -> the scene caption text. Default synthesis is the
    FIRST frame's caption (matches read_text's 'first legible' posture; the answer IS the caption)."""
    items = src.get("items", [])
    caps = [it["text"] for it in items if it.get("text")]
    if not caps:
        return None, None, None, "no-grounded-answer"   # nothing described -> honest ungrounded
    readout = caps[0]
    return readout, f"Scene: {readout}", None, None


def op_answer(step, env, cache):
    a = step["args"]
    src = env[a["from"]]
    q = a["question"]
    kind = src["kind"]
    if kind == "ordered":
        ans, verdict, yes, reason = _answer_temporal(src, q)
    elif kind == "number":
        ans, verdict, yes, reason = _answer_count(src, q)
    elif kind == "texts":
        ans, verdict, yes, reason = _answer_text(src, q)
    elif kind == "captions":
        ans, verdict, yes, reason = _answer_caption(src, q)
    elif kind in ("detections", "crops", "frames"):
        ans, verdict, yes, reason = _answer_presence(src, q)
    else:
        ans, verdict, yes, reason = None, None, None, "unsupported-kind"

    grounded = reason is None
    overlays, focus_ts = [], 0
    # Overlay the scorer box only on a grounded temporal answer; label is derived, not '#10'.
    if grounded and kind == "ordered":
        first = src["value"].get("first") or {}
        scorer = first.get("scorer_det")
        label = src["value"].get("subject_label") or "subject"
        if scorer:
            for ts in cache.analyzed_frames():
                for d in cache.detections_at(ts):
                    if d["det_id"] == scorer:
                        overlays.append({"box": d["box"], "label": f"{label} scored", "tone": "green", "kind": "box"})
                        focus_ts = ts

    if grounded:
        # Readout-style answers (number / texts / captions) render their answer string; yes/no
        # answers (ordered / presence) render the verdict. Keyed on `yes is None` so a new readout
        # kind (captions, codex P1) never falls through to "No" and discards its readout.
        output_label = ("Yes" if yes else "No") if yes is not None else ans
        result = {
            "id": step["id"], "op": "answer", "producer": "answer_question",
            "status": "done", "source": "live", "inputs": [a["from"]],
            "input_label": f"{kind} binding", "output_label": output_label,
            "confidence": None, "evidence": _evi(focus_ts, overlays),
        }
        binding = {"kind": "answer", "value": {
            "answer": ans, "verdict": verdict, "question": q, "yes": yes, "grounded": True,
        }}
        return binding, result

    # Ungrounded: the binding could not produce a grounded answer (Q1 -> A). No fake verdict.
    result = {
        "id": step["id"], "op": "answer", "producer": "answer_question",
        "status": "empty", "source": "live", "inputs": [a["from"]],
        "input_label": f"{kind} binding", "output_label": "(couldn't ground)",
        "confidence": None, "evidence": _evi(0, []),
        "note": f"could not ground an answer from a '{kind}' binding ({reason})",
    }
    binding = {"kind": "answer", "value": {
        "answer": None, "verdict": None, "question": q, "yes": None,
        "grounded": False, "reason": reason,
    }}
    return binding, result


OPS = {
    "sample_frames": op_sample_frames,
    "detect": op_detect,
    "describe_scene": op_describe_scene,
    "crop": op_crop,
    "read_text": op_read_text,
    "filter": op_filter,
    "count": op_count,
    "temporal_order": op_temporal_order,
    "answer": op_answer,
}

"""
Op implementations. Each is fn(step, env, cache) -> (binding, step_result):
  - binding: the value stored under step["id"] for later steps to reference.
  - step_result: the run-doc trace entry (status, provenance, EVIDENCE overlays).

Replay mode: detect/read_text/describe_scene read from the cache (real Azure outputs + pins);
crop/filter/count/temporal_order/answer are local compute. No live calls.

Binding kinds and item shapes:
  frames      items=[{frame_ts_ms}]
  detections  items=[{det_id,cls,confidence,box,frame_ts_ms}]
  captions    items=[{text,confidence,box,frame_ts_ms}]  (scene caption per sampled frame)
  crops       items=[{crop_id,det_id,box,person_box,person_conf,frame_ts_ms}]
  texts       items=[{det_id,text,confidence,source,note?,crop_box,person_box,frame_ts_ms}]
  number      value=int (op_count also carries frame_ts_ms/overlays, derived from its source
              items, so a downstream answer step inherits real evidence instead of ts 0)
  ordered     value={events,first,subject_dets,subject_label,subject_is_first_scorer}
  answer      value={answer,verdict,question,yes,grounded,reason?}
"""
from __future__ import annotations

from limits import (
    MAX_CAPTION_LEN,
    MAX_SAMPLED_FRAMES,
    MAX_SCENE_CAPTIONS,
    ON_PITCH_MIN_BOTTOM_Y,
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
    on_pitch = a.get("on_pitch", False)
    dets = []
    for f in frames:
        for d in cache.detections_at(f["frame_ts_ms"]):
            if d["cls"].lower() in classes:
                dets.append({**d, "frame_ts_ms": f["frame_ts_ms"]})
    raw_count = len(dets)
    if on_pitch:
        # Broadcast-framing heuristic (applied AFTER the class filter, BEFORE focus/overlay
        # derivation, so overlays/count only ever reflect kept detections): people standing on
        # the pitch have their box BOTTOM edge (y+h) in the lower portion of the frame; crowd/
        # camera operators/staff in the stands sit above the pitch horizon. See limits.py for the
        # threshold derivation against the real hero frame.
        dets = [d for d in dets if (d["box"]["y"] + d["box"]["h"]) >= ON_PITCH_MIN_BOTTOM_Y]
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
        if on_pitch and raw_count:
            # honest diagnostic-empty (D-DR5): detections existed but were all off-pitch, not "none found".
            result["note"] = (
                f"no on-pitch detections in the sampled frames ({raw_count} matched the class filter "
                "but all had box bottoms above the pitch horizon)"
            )
        else:
            result["note"] = "no detections in the sampled frames"
    return {"kind": "detections", "items": dets}, result


def op_describe_scene(step, env, cache):
    """Scene captioning (Azure Image Analysis CAPTION), replayed from the cache. One whole-image
    caption per sampled frame; the model free-text is truncated to MAX_CAPTION_LEN at the boundary
    (captions are model-generated, so bound the string BEFORE it enters a binding/answer/run-doc)
    and the kept count is capped at max_captions (default MAX_SCENE_CAPTIONS)."""
    a = step["args"]
    frames = env[a["frames"]]["items"]
    cap = a.get("max_captions", MAX_SCENE_CAPTIONS)
    items = []
    for f in frames:
        for c in cache.captions_at(f["frame_ts_ms"])[:1]:  # one whole-image caption per frame
            text = (c.get("text") or "")[:MAX_CAPTION_LEN]  # bound model free-text at the boundary
            items.append({**c, "text": text, "frame_ts_ms": f["frame_ts_ms"]})
    items = items[:cap]
    focus_ts = items[-1]["frame_ts_ms"] if items else (frames[-1]["frame_ts_ms"] if frames else 0)
    # CAPTION has no region; reuse the cached full-frame box (Box(0,0,1,1)) as a whole-scene overlay
    # (kind:"crop", tone:"teal"). EvidencePanel draws it as a full-frame outline; the caption text
    # is carried in output_label + the readout.
    overlays = [
        {"box": it.get("box", {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}),
         "label": it["text"], "confidence": it.get("confidence"), "tone": "teal", "kind": "crop"}
        for it in items if it["frame_ts_ms"] == focus_ts
    ]
    result = {
        "id": step["id"], "op": "describe_scene", "producer": "Azure AI Vision · caption",
        "status": "done" if items else "empty", "source": "cached",
        "inputs": [a["frames"]], "input_label": f"{len(frames)} frames",
        "output_label": (items[0]["text"] if items else "(no scene caption)"),
        "confidence": next((it["confidence"] for it in items if it.get("confidence") is not None), None),
        "evidence": _evi(focus_ts, overlays),
    }
    if not items:
        # diagnostic-empty (D-DR5): WHAT was searched + WHY, never a bare "no evidence".
        result["note"] = (
            f"no cached scene captions in the {len(frames)} sampled frame(s) — the describe_scene "
            "window/fps is outside the precomputed caption slice"
        )
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


def _collection_evidence(items, kind=None):
    """Derive a focus frame_ts_ms + sensible overlays from a frame-bearing items list (the shape
    detections/crops/texts/captions/frames all share), using the SAME focus convention every other
    op uses (the LAST item's frame). Returns (0, []) for an empty/frame-less list — the legitimate
    ts-0 case (no frame-derived input to show), never a silent black frame when items DO carry
    frame_ts_ms (the customer-visible bug this closes: op_count/op_answer previously hardcoded
    _evi(0, []) regardless of what their source items carried). Overlay styling mirrors the
    convention the PRODUCING op already uses for that kind (op_detect/op_read_text/op_crop/
    op_describe_scene) so the answer step's evidence looks like a continuation of the chain, not
    a different visual language."""
    if not items:
        return 0, []
    focus_ts = items[-1].get("frame_ts_ms", 0)
    overlays = []
    for it in items:
        if it.get("frame_ts_ms") != focus_ts:
            continue
        if kind == "captions":
            # mirrors op_describe_scene: whole-frame outline labeled with the caption text.
            box = it.get("box", {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0})
            overlays.append({"box": box, "label": it.get("text") or "(no caption)",
                              "confidence": it.get("confidence"), "tone": "teal", "kind": "crop"})
            continue
        if kind == "texts":
            # mirrors op_filter: the jersey/person box labeled '#<value>'.
            box = it.get("crop_box") or it.get("person_box") or it.get("box")
            if not box:
                continue
            val = it.get("text")
            overlays.append({"box": box, "label": f"#{val}" if val not in (None, "") else "read_text",
                              "confidence": it.get("confidence"), "tone": "teal", "kind": "box"})
            continue
        if kind == "crops":
            box = it.get("box") or it.get("person_box")
            if not box:
                continue
            overlays.append({"box": box, "label": "crop", "tone": "teal", "kind": "crop"})
            continue
        # detections (and any other collection carrying a plain box): mirrors op_detect.
        box = it.get("box")
        if not box:
            continue
        cls, conf = it.get("cls"), it.get("confidence")
        label = f"{cls} {conf:.2f}" if cls and conf is not None else (cls or "match")
        overlays.append({"box": box, "label": label, "confidence": conf, "tone": "azure", "kind": "box"})
    return focus_ts, overlays


def op_count(step, env, cache):
    a = step["args"]
    src = env[a["items"]]
    items = src["items"]
    n = len(items)
    # Count is a pure aggregate (no NEW frame of its own), but its INPUT is frame-derived — reuse
    # that focus instead of always showing ts 0 (the black-evidence-frame bug on count-shaped runs).
    focus_ts, overlays = _collection_evidence(items, kind=src["kind"])
    result = {
        "id": step["id"], "op": "count", "producer": "count", "status": "done", "source": "live",
        "inputs": [a["items"]], "input_label": src["kind"], "output_label": str(n),
        "confidence": None, "evidence": _evi(focus_ts, overlays),
    }
    # Carry the derived focus/overlays on the `number` binding (extra keys beyond kind/value; see
    # module docstring) so a downstream `answer` step reading a `number` binding — which has no
    # items of its own to derive evidence from — can inherit them instead of falling back to ts 0.
    return {"kind": "number", "value": n, "frame_ts_ms": focus_ts, "overlays": overlays}, result


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
    """scene: 'what is happening / describe the scene' -> the caption text. Default synthesis is
    the FIRST frame's caption (matches read_text's 'first legible' posture). An empty captions
    binding grounds out honestly (no fabricated description)."""
    items = src.get("items", [])
    caps = [it["text"] for it in items if it.get("text")]
    if not caps:
        return None, None, None, "no-grounded-answer"   # nothing captioned -> honest ground-out
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
    # The answer step is (almost always) the LAST step of a run, so a black/ts-0 evidence frame
    # here is the customer-visible "black evidence frame" bug. Every kind below has SOME
    # frame-derived lineage in its source binding when grounded; only a truly frame-less grounded
    # answer (there is none today, but future kinds might add one) legitimately keeps ts 0.
    if grounded and kind == "ordered":
        # Overlay the scorer box on a grounded temporal answer; label is derived, not '#10'.
        first = src["value"].get("first") or {}
        scorer = first.get("scorer_det")
        label = src["value"].get("subject_label") or "subject"
        if scorer:
            for ts in cache.analyzed_frames():
                for d in cache.detections_at(ts):
                    if d["det_id"] == scorer:
                        overlays.append({"box": d["box"], "label": f"{label} scored", "tone": "green", "kind": "box"})
                        focus_ts = ts
        if not overlays and first.get("ts_ms") is not None:
            # The scorer's det_id has no cached detection to draw a box from (or cache is bare),
            # but the goal event itself still carries a real frame timestamp — show THAT frame
            # rather than falling back to ts 0.
            focus_ts = first["ts_ms"]
    elif grounded and kind == "number":
        # `number` bindings (op_count) carry their derived focus/overlays as extra keys precisely
        # so this doesn't have to fall back to ts 0 just because `number` itself has no items.
        focus_ts = src.get("frame_ts_ms", 0)
        overlays = src.get("overlays", [])
    elif grounded and kind in ("texts", "captions", "detections", "crops", "frames"):
        focus_ts, overlays = _collection_evidence(src.get("items", []), kind=kind)

    if grounded:
        # Use the synthesized answer string for kinds whose answer IS free text/number (number,
        # texts, captions); the yes/no kinds render the verdict bit. A captions answer falling into
        # the yes/no branch would render as "No", discarding the caption readout (codex P1).
        output_label = ans if kind in ("number", "texts", "captions") else ("Yes" if yes else "No")
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

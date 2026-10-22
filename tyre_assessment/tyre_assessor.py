"""
Tyre Tread Assessment — OpenAI VLM (GPT-4o)
=============================================
Assesses tyre condition (tread depth, wear patterns, damage) from vehicle
photos using OpenAI's vision API. Acts as a professional tyre assessor.

Supported tyre positions:
  - front_left     : Front left tyre
  - front_right    : Front right tyre
  - back_left      : Back left tyre (rear left)
  - back_right     : Back right tyre (rear right)
  - spare          : Spare tyre

Usage:
  # Single tyre assessment
  python tyre_assessor.py --image front_left.jpg --position front_left --api-key sk-xxx

  # Assess multiple tyres
  python tyre_assessor.py --image fl.jpg fr.jpg bl.jpg br.jpg --position front_left front_right back_left back_right --api-key sk-xxx

  # Assess all positions (one image per position, in order)
  python tyre_assessor.py --image fl.jpg fr.jpg bl.jpg br.jpg spare.jpg --position all --api-key sk-xxx

  # Use env var for API key
  set OPENAI_API_KEY=sk-xxx
  python tyre_assessor.py --image front_left.jpg --position front_left
"""

import argparse
import base64
import cv2
import json
import os
import sys
import time
from openai import OpenAI
import mlflow


# ── Tyre Position Definitions ──────────────────────────────────────────────

POSITIONS = {
    "front_left": {"name": "Front Left Tyre", "abbrev": "FL"},
    "front_right": {"name": "Front Right Tyre", "abbrev": "FR"},
    "back_left": {"name": "Back Left Tyre", "abbrev": "BL"},
    "back_right": {"name": "Back Right Tyre", "abbrev": "BR"},
    "spare": {"name": "Spare Tyre", "abbrev": "SP"},
}


# ── Prompt ─────────────────────────────────────────────────────────────────

def build_prompt(position_key):
    pos = POSITIONS[position_key]
    is_spare = position_key == "spare"

    spare_extra = """
6. Is the spare tyre present at all? (it may be missing entirely)
7. If present, does it appear to be a full-size spare or a temporary/compact spare (space-saver)?""" if is_spare else ""

    spare_json_fields = """
  "spare_present": true/false,
  "spare_type": "full_size/compact_temporary/not_applicable",""" if is_spare else ""

    return f"""You are a certified vehicle tyre inspector conducting a formal tyre assessment for a ride-sharing vehicle compliance check. Your job is to determine whether this tyre PASSES or FAILS the safety inspection.

**Tyre being assessed: {pos['name']} ({pos['abbrev']})**

Carefully examine this tyre image and assess the following. Be thorough — a passenger's safety depends on your assessment.

── WHAT TO INSPECT ──

1. **TREAD DEPTH**: Look at the tread grooves (the channels cut into the rubber).
   - GOOD tread: deep, clearly visible grooves with well-defined edges
   - WORN tread: shallow grooves, tread bars becoming visible, smooth patches
   - BALD: no visible grooves, completely smooth surface, tread wear indicators are flush with the surface
   - Check if wear is EVEN across the tyre or UNEVEN (one side more worn than the other)

2. **TREAD PATTERN**: Check the pattern carved into the rubber.
   - Is the pattern still clearly visible and well-defined?
   - Are there any sections where the pattern has been worn away?

3. **SIDEWALL CONDITION**: Look at the side of the tyre (the flat wall between the rim and the tread).
   - Any bulges, bubbles, or blisters? (these indicate internal damage — FAIL)
   - Any cuts, cracks, or gouges?
   - Any exposed cords or wires? (immediate FAIL)

4. **SURFACE DAMAGE**: Look for visible damage on the tread surface.
   - Punctures, nails, screws, or foreign objects embedded in the tyre
   - Chunks of rubber missing
   - Deep cuts or slashes

5. **OVERALL APPEARANCE**: General condition assessment.
   - Does the tyre look aged/dry-rotted? (cracking in the rubber from age/sun)
   - Is the tyre properly inflated? (if visually determinable — flat or bulging)
   - Does the tyre appear to match the vehicle (correct size)?
{spare_extra}

── PASS / FAIL CRITERIA ──

PASS if ALL of these are true:
  - Tread depth appears adequate (grooves clearly visible, no bald spots)
  - No sidewall damage (no bulges, cuts, exposed cords)
  - No significant surface damage
  - Tyre appears properly inflated
  - Wear is reasonably even

FAIL if ANY of these are true:
  - Bald spots or tread wear indicators flush with surface
  - Sidewall bulges, cuts, or exposed cords
  - Foreign objects embedded (nails, screws)
  - Severe cracking or dry rot
  - Visibly flat or severely under/over-inflated
  - Severely uneven wear indicating alignment/suspension issues

Respond in this exact JSON format only, no other text:
{{
  "position": "{position_key}",{spare_json_fields}
  "result": "pass/fail",
  "tread_depth": "good/adequate/worn/bald",
  "tread_pattern_visible": true/false,
  "wear_evenness": "even/slightly_uneven/severely_uneven",
  "sidewall_condition": "good/minor_damage/major_damage",
  "surface_damage": "none/minor/major",
  "foreign_objects": true/false,
  "cracks_or_dry_rot": "none/minor/severe",
  "inflation": "normal/under_inflated/over_inflated/flat/cannot_determine",
  "confidence": "high/medium/low",
  "notes": "detailed observation — describe what you actually see on this specific tyre"
}}"""


# ── Core Functions ─────────────────────────────────────────────────────────

def image_to_base64(image_path, max_width=1024):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def assess_tyre(client, model, image_path, position_key):
    if position_key not in POSITIONS:
        raise ValueError(f"Unknown position: {position_key}. Available: {list(POSITIONS.keys())}")

    prompt = build_prompt(position_key)
    b64 = image_to_base64(image_path)

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
                    },
                ],
            }
        ],
        max_tokens=400,
        temperature=0,
    )

    api_duration = time.time() - start_time
    answer = response.choices[0].message.content.strip()

    try:
        if "```json" in answer:
            answer = answer.split("```json")[1].split("```")[0].strip()
        elif "```" in answer:
            answer = answer.split("```")[1].split("```")[0].strip()
        result = json.loads(answer)
    except (json.JSONDecodeError, IndexError):
        result = {
            "position": position_key,
            "result": "error",
            "confidence": "unknown",
            "notes": answer,
        }

    result["image"] = image_path
    result["api_duration_sec"] = round(api_duration, 2)
    return result


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vehicle Tyre Tread Assessment using OpenAI VLM")
    parser.add_argument("--image", nargs="+", required=True, help="Path(s) to tyre image(s)")
    parser.add_argument("--position", nargs="+", required=True,
                        help=f"Tyre position(s): {', '.join(POSITIONS.keys())}, or 'all'")
    parser.add_argument("--api-key", default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o",
                        help="OpenAI model (default: gpt-4o)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: tyre_results.json)")
    parser.add_argument("--mlflow-tracking-uri", default=None,
                        help="MLflow tracking URI (or set MLFLOW_TRACKING_URI env var)")
    parser.add_argument("--experiment", default="tyre-assessment-vlm",
                        help="MLflow experiment name (default: tyre-assessment-vlm)")
    parser.add_argument("--run-name", default=None,
                        help="Custom MLflow run name (e.g. 'good-tyres', 'bad-tyres')")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: provide --api-key or set OPENAI_API_KEY env var")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    positions_to_check = list(POSITIONS.keys()) if "all" in args.position else args.position

    for pos in positions_to_check:
        if pos not in POSITIONS:
            print(f"Error: unknown position '{pos}'")
            print(f"Available: {', '.join(POSITIONS.keys())}")
            sys.exit(1)

    for img in args.image:
        if not os.path.exists(img):
            print(f"Error: image not found: {img}")
            sys.exit(1)

    if len(args.image) != len(positions_to_check):
        print(f"Error: number of images ({len(args.image)}) must match number of positions ({len(positions_to_check)})")
        print("Each image corresponds to one tyre position, in order.")
        sys.exit(1)

    # ── MLflow Setup ────────────────────────────────────────────────────
    tracking_uri = args.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    print("=" * 60)
    print("  TYRE TREAD ASSESSMENT")
    print("=" * 60)
    print(f"  Tyres to assess: {len(positions_to_check)}")
    print(f"  Positions: {', '.join(positions_to_check)}")
    print(f"  Model: {args.model}")
    print(f"  MLflow experiment: {args.experiment}")
    print("=" * 60)

    all_results = []

    run_name = args.run_name or f"tyre-assess-{len(positions_to_check)}tyres"

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("model", args.model)
        mlflow.log_param("positions_checked", ",".join(positions_to_check))
        mlflow.log_param("num_tyres", len(positions_to_check))
        mlflow.log_param("image_names", ",".join(os.path.basename(img) for img in args.image))

        sample_prompt = build_prompt(positions_to_check[0])
        mlflow.log_text(sample_prompt, "prompt_template.txt")

        for idx, (img_path, pos_key) in enumerate(zip(args.image, positions_to_check)):
            pos_name = POSITIONS[pos_key]["name"]
            print(f"\n  [{idx+1}/{len(positions_to_check)}] Assessing {pos_name}")
            print(f"  Image: {os.path.basename(img_path)}")
            print(f"  {'-' * 50}")

            try:
                result = assess_tyre(client, args.model, img_path, pos_key)

                verdict = result.get("result", "?").upper()
                tread = result.get("tread_depth", "?")
                wear = result.get("wear_evenness", "?")
                sidewall = result.get("sidewall_condition", "?")
                confidence = result.get("confidence", "?")

                if verdict == "PASS":
                    icon = "PASS"
                elif verdict == "FAIL":
                    icon = "FAIL"
                else:
                    icon = "????"

                print(f"  Result:   [{icon}]")
                print(f"  Tread:    {tread} | Wear: {wear}")
                print(f"  Sidewall: {sidewall} | Confidence: {confidence}")
                if result.get("notes"):
                    print(f"  Notes:    {result['notes']}")

                mlflow.log_metric(f"pass_{pos_key}", 1 if verdict == "PASS" else 0)
                mlflow.log_metric(f"api_duration_{pos_key}", result.get("api_duration_sec", 0))

                tread_score = {"good": 3, "adequate": 2, "worn": 1, "bald": 0}.get(tread, -1)
                mlflow.log_metric(f"tread_score_{pos_key}", tread_score)

            except Exception as e:
                print(f"  ERROR: {e}")
                result = {"position": pos_key, "image": img_path, "result": "error", "error": str(e)}

            all_results.append(result)

        # ── Summary ─────────────────────────────────────────────────────
        print()
        print("=" * 60)
        print("  ASSESSMENT SUMMARY")
        print("=" * 60)

        pass_count = sum(1 for r in all_results if r.get("result") == "pass")
        fail_count = sum(1 for r in all_results if r.get("result") == "fail")
        total = len(all_results)

        for r in all_results:
            pos_key = r.get("position", "?")
            pos_name = POSITIONS.get(pos_key, {}).get("name", pos_key)
            verdict = r.get("result", "?").upper()
            tread = r.get("tread_depth", "?")
            icon = "PASS" if verdict == "PASS" else "FAIL" if verdict == "FAIL" else "????"
            print(f"  [{icon}] {pos_name:<22} Tread: {tread}")

        print(f"\n  Passed: {pass_count}/{total} | Failed: {fail_count}/{total}")

        if fail_count == 0:
            print("  OVERALL: ALL TYRES PASSED")
        else:
            print("  OVERALL: VEHICLE FAILED TYRE INSPECTION")

        print("=" * 60)

        mlflow.log_metric("total_pass", pass_count)
        mlflow.log_metric("total_fail", fail_count)
        mlflow.log_metric("total_assessed", total)
        mlflow.log_metric("pass_rate", round(pass_count / total * 100, 1) if total > 0 else 0)

        out_path = args.output or "tyre_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results saved: {out_path}")

        mlflow.log_artifact(out_path)

        print(f"  MLflow run logged successfully!")
        print(f"  View at: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()

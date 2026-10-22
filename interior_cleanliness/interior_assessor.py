"""
Interior Cleanliness Assessment — AWS Bedrock (Amazon Nova Lite)
================================================================
Assesses vehicle interior cleanliness from photos using AWS Bedrock's
vision API. Acts as a professional vehicle interior inspector for
ride-sharing compliance.

Usage:
  # Single image assessment
  python interior_assessor.py --image photo.jpg

  # Multiple images (different angles of the interior)
  python interior_assessor.py --image dashboard.jpg front_seats.jpg rear_seats.jpg floor.jpg

  # Custom model
  python interior_assessor.py --image photo.jpg --model amazon.nova-pro-v1:0

  # Custom run name for MLflow
  python interior_assessor.py --image photo.jpg --run-name "clean-interior-test"

Environment variables required:
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
  MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
"""

import argparse
import base64
import boto3
import cv2
import json
import os
import sys
import time
import mlflow


# ── Prompt ─────────────────────────────────────────────────────────────────

def build_prompt(image_name):
    return f"""You are a certified vehicle interior inspector conducting a formal cleanliness assessment for a ride-sharing vehicle compliance check. A passenger will sit in this vehicle next — your job is to determine whether the interior is clean enough for a paying customer.

**Image being assessed: {image_name}**

Carefully examine this interior photo and assess the following. Be thorough — a passenger's comfort and hygiene depends on your assessment.

── WHAT TO INSPECT ──

1. **SEATS**: Look at the seats visible in this image.
   - Any stains (food, drink, grease, ink, unknown)?
   - Any tears, rips, or holes in the upholstery?
   - Is the fabric/leather faded, worn, or peeling?
   - Any crumbs, hair, lint, or debris on the seats?

2. **FLOOR & MATS**: Look at the floor area and floor mats.
   - Any dirt, mud, sand, or dust accumulation?
   - Any trash, wrappers, bottles, or debris?
   - Are floor mats present and clean?
   - Any wet patches or liquid spills?

3. **DASHBOARD & CONSOLE**: Look at the dashboard, steering wheel, and center console.
   - Any dust buildup on surfaces?
   - Any sticky residue or spill marks?
   - Is the steering wheel clean or grimy?
   - Any clutter or personal items scattered around?

4. **DOORS & PANELS**: Look at door panels and interior trim.
   - Any scuff marks, scratches, or stains?
   - Are door handles clean?
   - Any damage to interior panels?

5. **CEILING / HEADLINER**: If visible, check the roof interior.
   - Any stains, sagging, or discoloration?
   - Any smoke stains or yellowing?

6. **ODOR INDICATORS**: Look for visual clues that suggest bad odor.
   - Visible mold or mildew anywhere?
   - Old food or drink containers?
   - Cigarette burns or ash residue?
   - Air freshener hanging (may indicate masking odor)?

7. **OVERALL IMPRESSION**: General cleanliness level.
   - Would a paying passenger feel comfortable sitting here?
   - Does the interior look professionally maintained or neglected?

── PASS / FAIL CRITERIA ──

PASS if ALL of these are true:
  - No visible stains on seats, floor, or surfaces
  - No trash or debris anywhere
  - Surfaces appear wiped/clean (minimal dust)
  - No tears or significant wear damage
  - No signs of mold, mildew, or smoke
  - Overall appears clean and well-maintained

FAIL if ANY of these are true:
  - Visible stains on seats or carpet
  - Trash, food waste, or debris present
  - Heavy dust or dirt accumulation
  - Mold, mildew, or cigarette burns
  - Torn or badly damaged upholstery
  - Overall looks neglected or unhygienic

Respond in this exact JSON format only, no other text:
{{
  "image": "{image_name}",
  "result": "pass/fail",
  "cleanliness_score": 1-10,
  "seats": {{
    "condition": "clean/minor_issues/dirty",
    "stains": true/false,
    "tears_or_damage": true/false,
    "debris": true/false
  }},
  "floor": {{
    "condition": "clean/minor_issues/dirty",
    "dirt_or_mud": true/false,
    "trash": true/false,
    "mats_present": true/false
  }},
  "dashboard_console": {{
    "condition": "clean/minor_issues/dirty",
    "dust_buildup": true/false,
    "clutter": true/false
  }},
  "odor_risk": "none/low/high",
  "confidence": "high/medium/low",
  "issues_found": ["list of specific issues found, empty if none"],
  "notes": "detailed observation — describe what you actually see in this image"
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


def assess_interior(bedrock_client, model, image_path):
    image_name = os.path.basename(image_path)
    prompt = build_prompt(image_name)
    b64 = image_to_base64(image_path)
    image_bytes = base64.b64decode(b64)

    start_time = time.time()
    response = bedrock_client.converse(
        modelId=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {"bytes": image_bytes},
                        }
                    },
                ],
            }
        ],
        inferenceConfig={"maxTokens": 500, "temperature": 0},
    )

    api_duration = time.time() - start_time
    answer = response["output"]["message"]["content"][0]["text"].strip()

    try:
        if "```json" in answer:
            answer = answer.split("```json")[1].split("```")[0].strip()
        elif "```" in answer:
            answer = answer.split("```")[1].split("```")[0].strip()
        result = json.loads(answer)
    except (json.JSONDecodeError, IndexError):
        result = {
            "image": image_name,
            "result": "error",
            "cleanliness_score": 0,
            "confidence": "unknown",
            "notes": answer,
        }

    result["image_path"] = image_path
    result["api_duration_sec"] = round(api_duration, 2)
    return result


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vehicle Interior Cleanliness Assessment using AWS Bedrock VLM")
    parser.add_argument("--image", nargs="+", required=True, help="Path(s) to interior image(s)")
    parser.add_argument("--model", default="apac.amazon.nova-lite-v1:0",
                        help="Bedrock model ID (default: apac.amazon.nova-lite-v1:0)")
    parser.add_argument("--region", default=None,
                        help="AWS region (or set AWS_DEFAULT_REGION env var)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: interior_results.json)")
    parser.add_argument("--mlflow-tracking-uri", default=None,
                        help="MLflow tracking URI (or set MLFLOW_TRACKING_URI env var)")
    parser.add_argument("--experiment", default="interior-cleanliness-vlm",
                        help="MLflow experiment name (default: interior-cleanliness-vlm)")
    parser.add_argument("--run-name", default=None,
                        help="Custom MLflow run name (e.g. 'clean-interior', 'dirty-interior')")
    args = parser.parse_args()

    region = args.region or os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)

    for img in args.image:
        if not os.path.exists(img):
            print(f"Error: image not found: {img}")
            sys.exit(1)

    tracking_uri = args.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    run_name = args.run_name or f"interior-assess-{len(args.image)}img"

    print("=" * 60)
    print("  INTERIOR CLEANLINESS ASSESSMENT")
    print("=" * 60)
    print(f"  Images: {len(args.image)}")
    print(f"  Model: {args.model}")
    print(f"  Provider: AWS Bedrock")
    print(f"  Region: {region}")
    print(f"  MLflow experiment: {args.experiment}")
    print("=" * 60)

    all_results = []

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("model", args.model)
        mlflow.log_param("provider", "aws-bedrock")
        mlflow.log_param("region", region)
        mlflow.log_param("num_images", len(args.image))
        mlflow.log_param("image_names", ",".join(os.path.basename(img) for img in args.image))

        sample_prompt = build_prompt("sample.jpg")
        mlflow.log_text(sample_prompt, "prompt_template.txt")

        for idx, img_path in enumerate(args.image):
            img_name = os.path.basename(img_path)
            print(f"\n  [{idx+1}/{len(args.image)}] Assessing: {img_name}")
            print(f"  {'-' * 50}")

            try:
                result = assess_interior(bedrock_client, args.model, img_path)

                verdict = result.get("result", "?").upper()
                score = result.get("cleanliness_score", "?")
                confidence = result.get("confidence", "?")
                issues = result.get("issues_found", [])

                if verdict == "PASS":
                    icon = "PASS"
                elif verdict == "FAIL":
                    icon = "FAIL"
                else:
                    icon = "????"

                print(f"  Result:     [{icon}]")
                print(f"  Score:      {score}/10")
                print(f"  Confidence: {confidence}")

                if issues:
                    print(f"  Issues:")
                    for issue in issues:
                        print(f"    - {issue}")

                if result.get("notes"):
                    print(f"  Notes:      {result['notes']}")

                tag = os.path.splitext(img_name)[0]
                mlflow.log_metric(f"pass_{tag}", 1 if verdict == "PASS" else 0)
                mlflow.log_metric(f"score_{tag}", score if isinstance(score, (int, float)) else 0)
                mlflow.log_metric(f"api_duration_{tag}", result.get("api_duration_sec", 0))

            except Exception as e:
                print(f"  ERROR: {e}")
                result = {"image": img_name, "image_path": img_path, "result": "error", "error": str(e)}

            all_results.append(result)

        # ── Summary ─────────────────────────────────────────────────────
        print()
        print("=" * 60)
        print("  ASSESSMENT SUMMARY")
        print("=" * 60)

        pass_count = sum(1 for r in all_results if r.get("result") == "pass")
        fail_count = sum(1 for r in all_results if r.get("result") == "fail")
        total = len(all_results)
        scores = [r.get("cleanliness_score", 0) for r in all_results if isinstance(r.get("cleanliness_score"), (int, float))]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0

        for r in all_results:
            img_name = os.path.basename(r.get("image_path", r.get("image", "?")))
            verdict = r.get("result", "?").upper()
            score = r.get("cleanliness_score", "?")
            icon = "PASS" if verdict == "PASS" else "FAIL" if verdict == "FAIL" else "????"
            print(f"  [{icon}] {img_name:<30} Score: {score}/10")

        print(f"\n  Average Score: {avg_score}/10")
        print(f"  Passed: {pass_count}/{total} | Failed: {fail_count}/{total}")

        if fail_count == 0:
            print("  OVERALL: INTERIOR PASSED CLEANLINESS CHECK")
        else:
            print("  OVERALL: INTERIOR FAILED CLEANLINESS CHECK")

        print("=" * 60)

        mlflow.log_metric("total_pass", pass_count)
        mlflow.log_metric("total_fail", fail_count)
        mlflow.log_metric("total_assessed", total)
        mlflow.log_metric("avg_cleanliness_score", avg_score)
        mlflow.log_metric("pass_rate", round(pass_count / total * 100, 1) if total > 0 else 0)

        out_path = args.output or "interior_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results saved: {out_path}")

        mlflow.log_artifact(out_path)

        print(f"  MLflow run logged successfully!")
        print(f"  View at: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()

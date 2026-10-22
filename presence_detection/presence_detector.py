"""
Presence Detection — AWS Bedrock (Amazon Nova Lite)
====================================================
Detects whether a specific item is present in a vehicle photo
using AWS Bedrock's vision API. One common function for all items.

Supported items:
  - cng_tank         : CNG tank (CNG vehicles only)
  - medical_kit      : Medical / first aid kit
  - fire_extinguisher: Fire extinguisher
  - triangle_reflector: Triangle reflector (folded format)
  - tool_kit         : Tool kit (jack and wrench)
  - seat_belt        : Seat belts (driver, rear and 3rd row)
  - mobile_holder    : Mobile phone holder

Usage:
  # Single item check
  python presence_detector.py --image photo.jpg --item medical_kit

  # Check all items
  python presence_detector.py --image photo.jpg --item all

  # Multiple images, all items
  python presence_detector.py --image img1.jpg img2.jpg --item all

  # Custom run name for MLflow
  python presence_detector.py --image photo.jpg --item all --run-name "bedrock-test"

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


# ── Item Definitions ────────────────────────────────────────────────────────

ITEMS = {
    "cng_tank": {
        "name": "CNG Tank",
        "description": "a CNG (Compressed Natural Gas) tank or cylinder, typically mounted in the trunk or under the vehicle",
        "look_for": "a cylindrical metal tank, usually silver/grey, with valves and fittings",
    },
    "medical_kit": {
        "name": "Medical Kit",
        "description": "a medical or first aid kit",
        "look_for": "a box or bag marked with a red cross/plus symbol, or labeled as first aid kit, usually red or green colored",
    },
    "fire_extinguisher": {
        "name": "Fire Extinguisher",
        "description": "a fire extinguisher",
        "look_for": "a cylindrical red canister with a nozzle/hose and pressure gauge, typically small vehicle-sized",
    },
    "triangle_reflector": {
        "name": "Triangle Reflector",
        "description": "a warning triangle reflector in folded format",
        "look_for": "a folded triangular warning sign with reflective orange/red surface, may be in a plastic case",
    },
    "tool_kit": {
        "name": "Tool Kit",
        "description": "a vehicle tool kit containing a jack and wrench",
        "look_for": "a jack (scissor jack or bottle jack), a wheel wrench/lug wrench, possibly in a bag or case",
    },
    "seat_belt": {
        "name": "Seat Belt",
        "description": "seat belts in the vehicle (driver seat, rear seats, and 3rd row if applicable)",
        "look_for": "seat belt straps, buckles, and retractors on the seats. Check if they are present and appear functional",
    },
    "mobile_holder": {
        "name": "Mobile Holder",
        "description": "a mobile phone holder/mount installed in the vehicle",
        "look_for": "a phone mount or cradle attached to the dashboard, windshield, or air vent",
    },
}


# ── Core Function ───────────────────────────────────────────────────────────

def build_prompt(item_key):
    item = ITEMS[item_key]
    return f"""You are inspecting a vehicle photo to check if a specific item is present.

Item to check: **{item['name']}**
Description: {item['description']}
What to look for: {item['look_for']}

Carefully examine the image and determine:
1. Is the {item['name']} visible/present in this image?
2. If present, what is its condition (good/damaged/unclear)?
3. How confident are you?

Respond in this exact JSON format only, no other text:
{{"item": "{item_key}", "present": true/false, "condition": "good/damaged/unclear/not_applicable", "confidence": "high/medium/low", "notes": "brief observation about what you see"}}"""


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


def check_presence(bedrock_client, model, image_path, item_key):
    if item_key not in ITEMS:
        raise ValueError(f"Unknown item: {item_key}. Available: {list(ITEMS.keys())}")

    prompt = build_prompt(item_key)
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
        inferenceConfig={"maxTokens": 200, "temperature": 0},
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
            "item": item_key,
            "present": None,
            "condition": "unknown",
            "confidence": "unknown",
            "notes": answer,
        }

    result["image"] = image_path
    result["api_duration_sec"] = round(api_duration, 2)
    return result


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vehicle Item Presence Detection using AWS Bedrock VLM")
    parser.add_argument("--image", nargs="+", required=True, help="Path(s) to image file(s)")
    parser.add_argument("--item", nargs="+", required=True,
                        help=f"Item(s) to check: {', '.join(ITEMS.keys())}, or 'all'")
    parser.add_argument("--model", default="apac.amazon.nova-lite-v1:0",
                        help="Bedrock model ID (default: apac.amazon.nova-lite-v1:0)")
    parser.add_argument("--region", default=None,
                        help="AWS region (or set AWS_DEFAULT_REGION env var)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: presence_results.json)")
    parser.add_argument("--mlflow-tracking-uri", default=None,
                        help="MLflow tracking URI (or set MLFLOW_TRACKING_URI env var)")
    parser.add_argument("--experiment", default="presence-detection-vlm",
                        help="MLflow experiment name (default: presence-detection-vlm)")
    parser.add_argument("--run-name", default=None,
                        help="Custom MLflow run name")
    args = parser.parse_args()

    region = args.region or os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)

    items_to_check = list(ITEMS.keys()) if "all" in args.item else args.item

    for item_key in items_to_check:
        if item_key not in ITEMS:
            print(f"Error: unknown item '{item_key}'")
            print(f"Available: {', '.join(ITEMS.keys())}")
            sys.exit(1)

    for img in args.image:
        if not os.path.exists(img):
            print(f"Error: image not found: {img}")
            sys.exit(1)

    total_checks = len(args.image) * len(items_to_check)

    tracking_uri = args.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    run_name = args.run_name or f"presence-check-{len(args.image)}img"

    print("=" * 60)
    print("  VEHICLE ITEM PRESENCE DETECTION")
    print("=" * 60)
    print(f"  Images: {len(args.image)}")
    print(f"  Items:  {', '.join(items_to_check)}")
    print(f"  Model:  {args.model}")
    print(f"  Provider: AWS Bedrock")
    print(f"  Region: {region}")
    print(f"  Total API calls: {total_checks}")
    print(f"  MLflow experiment: {args.experiment}")
    print("=" * 60)

    all_results = []
    check_num = 0

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("model", args.model)
        mlflow.log_param("provider", "aws-bedrock")
        mlflow.log_param("region", region)
        mlflow.log_param("items_checked", ",".join(items_to_check))
        mlflow.log_param("num_images", len(args.image))
        mlflow.log_param("total_api_calls", total_checks)

        sample_prompt = build_prompt(items_to_check[0])
        mlflow.log_text(sample_prompt, "prompt_template.txt")

        for img_path in args.image:
            print(f"\n  Image: {os.path.basename(img_path)}")
            print(f"  {'-' * 50}")

            for item_key in items_to_check:
                check_num += 1
                item_name = ITEMS[item_key]["name"]
                print(f"  [{check_num}/{total_checks}] Checking {item_name} ... ", end="", flush=True)

                try:
                    result = check_presence(bedrock_client, args.model, img_path, item_key)

                    present = result.get("present")
                    condition = result.get("condition", "?")
                    confidence = result.get("confidence", "?")

                    if present is True:
                        status = f"PRESENT ({condition}, conf: {confidence})"
                    elif present is False:
                        status = f"NOT PRESENT (conf: {confidence})"
                    else:
                        status = f"UNCERTAIN (conf: {confidence})"

                    print(status)
                    if result.get("notes"):
                        print(f"    {result['notes']}")

                    tag = f"{os.path.basename(img_path)}_{item_key}"
                    mlflow.log_metric(f"present_{tag}", 1 if present else 0)
                    mlflow.log_metric(f"api_duration_{tag}", result.get("api_duration_sec", 0))

                except Exception as e:
                    print(f"ERROR: {e}")
                    result = {"item": item_key, "image": img_path, "error": str(e)}

                all_results.append(result)

        # ── Summary ─────────────────────────────────────────────────────
        print()
        print("=" * 60)
        print("  SUMMARY")
        print("=" * 60)

        present_count = sum(1 for r in all_results if r.get("present") is True)
        absent_count = sum(1 for r in all_results if r.get("present") is False)

        for img_path in args.image:
            img_results = [r for r in all_results if r.get("image") == img_path]
            print(f"\n  {os.path.basename(img_path)}:")

            for r in img_results:
                item_name = ITEMS.get(r.get("item", ""), {}).get("name", r.get("item", "?"))
                present = r.get("present")
                if present is True:
                    icon = "PASS"
                elif present is False:
                    icon = "FAIL"
                else:
                    icon = "????"
                print(f"    [{icon}] {item_name}")

        print()
        print("=" * 60)

        mlflow.log_metric("total_present", present_count)
        mlflow.log_metric("total_absent", absent_count)
        mlflow.log_metric("total_checks", total_checks)

        out_path = args.output or "presence_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results saved: {out_path}")

        mlflow.log_artifact(out_path)

        print(f"  MLflow run logged successfully!")
        print(f"  View at: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()

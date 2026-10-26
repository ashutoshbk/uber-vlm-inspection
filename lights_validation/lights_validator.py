"""
Lights Validation — AWS Bedrock (Amazon Nova Lite)
===================================================
Validates vehicle headlamps and tail lamps by analyzing video frames
using AWS Bedrock's vision API. Checks if lights are functioning
(turning on/off or staying on) across consecutive frames.

Supported light types:
  - headlamp  : Front headlights (low beam, high beam)
  - taillamp  : Rear tail lights (brake lights, running lights)

Logic:
  1. Extract frames from video at regular intervals (default: every 1 second)
  2. Send each frame to the VLM to check if the light is ON or OFF
  3. If light is ON in at least one frame → light is functioning (PASS)
  4. If light is OFF in all frames → light is not working (FAIL)

Usage:
  # Check headlamp video
  python lights_validator.py --video headlamp.mp4 --type headlamp

  # Check tail lamp video
  python lights_validator.py --video taillamp.mp4 --type taillamp

  # Custom interval
  python lights_validator.py --video headlamp.mp4 --type headlamp --interval 0.5

  # Custom run name for MLflow
  python lights_validator.py --video headlamp.mp4 --type headlamp --run-name "headlamp-test"

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


LIGHT_TYPES = {
    "headlamp": {
        "name": "Headlamp",
        "description": "front headlights of the vehicle",
        "look_for": "front headlights (low beam or high beam) — check if they are illuminated, glowing, or casting light",
    },
    "taillamp": {
        "name": "Tail Lamp",
        "description": "rear tail lights of the vehicle",
        "look_for": "rear tail lights, brake lights, or running lights — check if they are illuminated, glowing red, or casting light",
    },
}


def build_prompt(light_type, frame_num, timestamp):
    lt = LIGHT_TYPES[light_type]
    return f"""You are a certified vehicle inspector checking if the vehicle's {lt['description']} are functioning.

**Light type: {lt['name']}**
**Frame: #{frame_num} (t={timestamp}s)**

Carefully examine this video frame and determine:

1. **VISIBILITY**: Can you see the {lt['name']}(s) in this frame?
   - Look for: {lt['look_for']}

2. **STATUS**: Are the lights ON or OFF?
   - ON: lights are visibly glowing, illuminated, or you can see light being emitted/reflected
   - OFF: lights are dark, not glowing, no visible illumination

3. **CONDITION**: If ON, do the lights appear to be working normally?
   - Both sides working (left and right)?
   - Brightness appears normal (not dim or flickering)?
   - Any visible damage to the light housing?

4. **COUNT**: How many individual lights can you see that are ON?

Respond in this exact JSON format only, no other text:
{{
  "light_type": "{light_type}",
  "frame": {frame_num},
  "timestamp": {timestamp},
  "lights_visible": true/false,
  "lights_on": true/false,
  "left_working": true/false,
  "right_working": true/false,
  "brightness": "normal/dim/bright/off",
  "damage_visible": true/false,
  "lights_on_count": 0,
  "confidence": "high/medium/low",
  "notes": "brief observation about what you see"
}}"""


def frame_to_bytes(frame, max_width=1024):
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return bytes(buffer)


def extract_frames(video_path, interval_sec=1.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    frame_interval = max(1, int(fps * interval_sec))

    print(f"  Video:    {video_path}")
    print(f"  Duration: {duration:.1f}s | FPS: {fps:.0f}")
    print(f"  Interval: every {interval_sec}s")

    frames = []
    timestamps = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_interval == 0:
            frames.append(frame)
            timestamps.append(round(frame_idx / fps, 1))

        frame_idx += 1

    cap.release()
    print(f"  Extracted: {len(frames)} frames")
    return frames, timestamps


def analyze_frame(bedrock_client, model, frame, light_type, frame_num, timestamp):
    prompt = build_prompt(light_type, frame_num, timestamp)
    image_bytes = frame_to_bytes(frame)

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
        inferenceConfig={"maxTokens": 300, "temperature": 0},
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
            "light_type": light_type,
            "frame": frame_num,
            "timestamp": timestamp,
            "lights_visible": None,
            "lights_on": None,
            "confidence": "unknown",
            "notes": answer,
        }

    result["api_duration_sec"] = round(api_duration, 2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Vehicle Lights Validation using AWS Bedrock VLM")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--type", required=True, choices=list(LIGHT_TYPES.keys()),
                        help="Light type to check: headlamp or taillamp")
    parser.add_argument("--model", default="apac.amazon.nova-lite-v1:0",
                        help="Bedrock model ID (default: apac.amazon.nova-lite-v1:0)")
    parser.add_argument("--region", default=None,
                        help="AWS region (or set AWS_DEFAULT_REGION env var)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Frame extraction interval in seconds (default: 1.0)")
    parser.add_argument("--save", action="store_true",
                        help="Save extracted frames to disk")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    parser.add_argument("--mlflow-tracking-uri", default=None,
                        help="MLflow tracking URI (or set MLFLOW_TRACKING_URI env var)")
    parser.add_argument("--experiment", default="lights-validation-vlm",
                        help="MLflow experiment name (default: lights-validation-vlm)")
    parser.add_argument("--run-name", default=None,
                        help="Custom MLflow run name")
    args = parser.parse_args()

    region = args.region or os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)

    if not os.path.exists(args.video):
        print(f"Error: video not found: {args.video}")
        sys.exit(1)

    light_type = args.type
    light_name = LIGHT_TYPES[light_type]["name"]

    tracking_uri = args.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    print("=" * 60)
    print(f"  LIGHTS VALIDATION — {light_name.upper()}")
    print("=" * 60)
    print(f"  Light type: {light_name}")
    print(f"  Model:      {args.model}")
    print(f"  Provider:   AWS Bedrock")
    print(f"  Region:     {region}")

    frames, timestamps = extract_frames(args.video, args.interval)

    if len(frames) < 1:
        print("Error: no frames extracted. Check the video file.")
        sys.exit(1)

    if args.save:
        frames_dir = os.path.splitext(args.video)[0] + "_frames"
        os.makedirs(frames_dir, exist_ok=True)
        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            cv2.imwrite(os.path.join(frames_dir, f"frame_{i:03d}_t{ts}s.jpg"), frame)
        print(f"  Frames saved to: {frames_dir}/")

    print(f"\n  Analyzing {len(frames)} frames...")
    print("-" * 60)

    run_name = args.run_name or f"{light_type}-validate-{len(frames)}frames"

    frame_results = []
    lights_on_count = 0
    lights_visible_count = 0

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("model", args.model)
        mlflow.log_param("provider", "aws-bedrock")
        mlflow.log_param("region", region)
        mlflow.log_param("light_type", light_type)
        mlflow.log_param("video", os.path.basename(args.video))
        mlflow.log_param("interval_sec", args.interval)
        mlflow.log_param("total_frames", len(frames))

        sample_prompt = build_prompt(light_type, 1, 0.0)
        mlflow.log_text(sample_prompt, "prompt_template.txt")

        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            frame_num = i + 1
            print(f"  Frame {frame_num}/{len(frames)} (t={ts}s) ... ", end="", flush=True)

            try:
                result = analyze_frame(bedrock_client, args.model, frame, light_type, frame_num, ts)

                lights_on = result.get("lights_on")
                lights_visible = result.get("lights_visible")
                brightness = result.get("brightness", "?")
                confidence = result.get("confidence", "?")

                if lights_visible:
                    lights_visible_count += 1
                if lights_on:
                    lights_on_count += 1

                if lights_on:
                    status = f"ON (brightness: {brightness}, conf: {confidence})"
                elif lights_visible:
                    status = f"OFF (conf: {confidence})"
                else:
                    status = f"NOT VISIBLE (conf: {confidence})"

                print(status)
                if result.get("notes"):
                    print(f"    {result['notes']}")

                mlflow.log_metric(f"lights_on_frame{frame_num}", 1 if lights_on else 0)
                mlflow.log_metric(f"api_duration_frame{frame_num}", result.get("api_duration_sec", 0))

            except Exception as e:
                print(f"ERROR: {e}")
                result = {"light_type": light_type, "frame": frame_num, "timestamp": ts, "error": str(e)}

            frame_results.append(result)

        # ── Summary ─────────────────────────────────────────────────────
        print()
        print("=" * 60)
        print("  RESULTS")
        print("=" * 60)
        total = len(frames)
        print(f"  Light type:       {light_name}")
        print(f"  Total frames:     {total}")
        print(f"  Lights visible:   {lights_visible_count}/{total}")
        print(f"  Lights ON:        {lights_on_count}/{total}")

        if lights_visible_count == 0:
            verdict = "FAIL"
            reason = "Lights not visible in any frame"
        elif lights_on_count > 0:
            verdict = "PASS"
            reason = f"{light_name} functioning — ON in {lights_on_count}/{total} frames"
        else:
            verdict = "FAIL"
            reason = f"{light_name} not working — visible but OFF in all frames"

        print(f"\n  VERDICT: [{verdict}] {reason}")
        print("=" * 60)

        mlflow.log_metric("lights_visible_count", lights_visible_count)
        mlflow.log_metric("lights_on_count", lights_on_count)
        mlflow.log_metric("lights_on_rate", round(lights_on_count / total * 100, 1) if total > 0 else 0)
        mlflow.log_metric("pass", 1 if verdict == "PASS" else 0)

        output = {
            "video": args.video,
            "light_type": light_type,
            "model": args.model,
            "provider": "aws-bedrock",
            "region": region,
            "interval_sec": args.interval,
            "total_frames": total,
            "lights_visible_count": lights_visible_count,
            "lights_on_count": lights_on_count,
            "verdict": verdict,
            "reason": reason,
            "frames": frame_results,
        }

        out_path = args.output or f"{light_type}_results.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved: {out_path}")

        mlflow.log_artifact(out_path)

        print(f"  MLflow run logged successfully!")
        print(f"  View at: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()

"""
Wiper Movement Detection — OpenAI VLM (GPT-4o)
================================================
Detects if a car windshield wiper is actively wiping by comparing
consecutive video frames using OpenAI's vision API.

Logic:
  1. Extract frames from video at regular intervals (default: every 2 seconds)
  2. Send pairs of consecutive frames to GPT-4o
  3. Ask if the wiper position changed between the two frames
  4. If wiper moves in majority of pairs → wiper is functioning

Usage:
  python wiper_detector.py --video wiper_video.mp4 --api-key sk-xxx
  python wiper_detector.py --video wiper_video.mp4 --api-key sk-xxx --interval 3
  python wiper_detector.py --video wiper_video.mp4 --interval 1 --save
"""

import argparse
import base64
import cv2
import json
import os
import sys
import time
from openai import OpenAI


PROMPT = """You are analyzing two consecutive frames from a car dashcam or surveillance video, taken a few seconds apart.

Look at the windshield wiper(s) in both images carefully.

Answer these questions:
1. Can you see the windshield wiper(s) in the images?
2. Is the wiper in a DIFFERENT position between Frame 1 and Frame 2?
3. Does it appear the wiper is actively wiping/moving?

Respond in this exact JSON format only, no other text:
{"wiper_visible": true/false, "position_changed": true/false, "wiper_active": true/false, "confidence": "high/medium/low", "notes": "brief observation"}"""


def extract_frames(video_path, interval_sec=2.0):
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


def frame_to_base64(frame, max_width=800):
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def analyze_pair(client, model, frame1, frame2, t1, t2):
    b64_1 = frame_to_base64(frame1)
    b64_2 = frame_to_base64(frame2)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Frame 1 (t={t1}s) and Frame 2 (t={t2}s):\n\n{PROMPT}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_1}", "detail": "low"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_2}", "detail": "low"},
                    },
                ],
            }
        ],
        max_tokens=200,
        temperature=0,
    )

    answer = response.choices[0].message.content.strip()
    try:
        if "```json" in answer:
            answer = answer.split("```json")[1].split("```")[0].strip()
        elif "```" in answer:
            answer = answer.split("```")[1].split("```")[0].strip()
        result = json.loads(answer)
    except (json.JSONDecodeError, IndexError):
        result = {
            "wiper_visible": None,
            "position_changed": None,
            "wiper_active": None,
            "confidence": "unknown",
            "notes": answer,
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Wiper Movement Detection using OpenAI VLM")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--api-key", default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o",
                        help="OpenAI model (default: gpt-4o)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Frame extraction interval in seconds (default: 2.0)")
    parser.add_argument("--save", action="store_true",
                        help="Save extracted frames to disk")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <video>_wiper_results.json)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: provide --api-key or set OPENAI_API_KEY env var")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print("=" * 60)
    print("  WIPER MOVEMENT DETECTION")
    print("=" * 60)

    frames, timestamps = extract_frames(args.video, args.interval)

    if len(frames) < 2:
        print("Error: need at least 2 frames. Try a shorter interval.")
        sys.exit(1)

    if args.save:
        frames_dir = os.path.splitext(args.video)[0] + "_frames"
        os.makedirs(frames_dir, exist_ok=True)
        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            cv2.imwrite(os.path.join(frames_dir, f"frame_{i:03d}_t{ts}s.jpg"), frame)
        print(f"  Frames saved to: {frames_dir}/")

    num_pairs = len(frames) - 1
    print(f"\n  Analyzing {num_pairs} frame pairs with {args.model}...")
    print("-" * 60)

    pair_results = []
    active_count = 0
    visible_count = 0

    for i in range(num_pairs):
        t1, t2 = timestamps[i], timestamps[i + 1]
        print(f"  Pair {i+1}/{num_pairs}: t={t1}s vs t={t2}s ... ", end="", flush=True)

        try:
            result = analyze_pair(client, args.model, frames[i], frames[i + 1], t1, t2)
            result["pair"] = i + 1
            result["t1"] = t1
            result["t2"] = t2

            status = []
            if result.get("wiper_visible"):
                visible_count += 1
                status.append("visible")
            if result.get("wiper_active"):
                active_count += 1
                status.append("ACTIVE")
            if result.get("position_changed"):
                status.append("moved")

            conf = result.get("confidence", "?")
            print(f"[{', '.join(status) or 'no wiper'}] (conf: {conf})")

            if result.get("notes"):
                print(f"    {result['notes']}")

        except Exception as e:
            print(f"ERROR: {e}")
            result = {"pair": i + 1, "t1": t1, "t2": t2, "error": str(e)}

        pair_results.append(result)

    # ── Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Total pairs analyzed:  {num_pairs}")
    print(f"  Wiper visible:         {visible_count}/{num_pairs}")
    print(f"  Wiper active:          {active_count}/{num_pairs}")

    if visible_count == 0:
        verdict = "NO WIPER DETECTED"
    elif active_count > num_pairs * 0.5:
        verdict = "WIPER IS FUNCTIONING — active in majority of frames"
    elif active_count > 0:
        verdict = "WIPER PARTIALLY ACTIVE — intermittent movement detected"
    else:
        verdict = "WIPER NOT MOVING — no movement detected"

    print(f"\n  VERDICT: {verdict}")
    print("=" * 60)

    # ── Save results ────────────────────────────────────────────────────
    output = {
        "video": args.video,
        "model": args.model,
        "interval_sec": args.interval,
        "total_pairs": num_pairs,
        "wiper_visible_count": visible_count,
        "wiper_active_count": active_count,
        "verdict": verdict,
        "pairs": pair_results,
    }

    out_path = args.output or (os.path.splitext(args.video)[0] + "_wiper_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()

# Wiper Movement Detection

Detects if windshield wipers are actively wiping by analyzing consecutive video frames using AWS Bedrock VLM.

**Status:** Tested and working.

## How It Works

1. Extracts frames from video at regular intervals (default: every 1 second)
2. Sends pairs of consecutive frames to the VLM
3. Asks if wiper position changed between frames
4. If wiper position changed in at least one pair → wiper is functioning (PASS)

## Usage

```bash
# Basic usage
python wiper_detector.py --video wiper_video.mp4

# Custom interval (check every 1 second)
python wiper_detector.py --video wiper_video.mp4 --interval 1

# Save extracted frames to disk
python wiper_detector.py --video wiper_video.mp4 --save

# Custom run name for MLflow
python wiper_detector.py --video wiper_video.mp4 --run-name "wiper-test"
```

## Output

Returns: `verdict` (functioning / not moving / no wiper detected), per-pair analysis with `wiper_visible`, `position_changed`, `wiper_active`.

## Verdicts

| Verdict | Meaning |
|---------|---------|
| WIPER IS FUNCTIONING | Position changed in at least one frame pair |
| WIPER NOT MOVING | Visible but no movement in any pair |
| NO WIPER DETECTED | Not visible in any frames |

## MLflow

- **Experiment:** `wiper-detection-vlm`
- **Params:** model, provider, region, video name, interval, total pairs
- **Metrics:** wiper visible/active per pair, active rate %
- **Artifacts:** results JSON, prompt template

## Environment Variables

```bash
# AWS credentials (stored in conda env, never in code)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=ap-south-1

# MLflow
MLFLOW_TRACKING_URI=your_mlflow_url
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password
```

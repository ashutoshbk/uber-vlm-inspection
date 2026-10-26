# Lights Validation

Validates vehicle headlamps and tail lamps by analyzing video frames using AWS Bedrock VLM. Checks if lights are functioning (ON/OFF) across consecutive frames.

## Supported Light Types

| Light Type | Key | What it checks |
|------------|-----|----------------|
| Headlamp | `headlamp` | Front headlights (low beam, high beam) |
| Tail Lamp | `taillamp` | Rear tail lights (brake lights, running lights) |

## How It Works

1. Extracts frames from video at regular intervals (default: every 1 second)
2. Sends each frame to the VLM to check if the light is ON or OFF
3. If light is ON in at least one frame → PASS (functioning, light is working)
4. If light is OFF in all frames → FAIL (not working)

## Usage

```bash
# Check headlamp
python lights_validator.py --video test_videos/head_lamp_video.mp4 --type headlamp

# Check tail lamp
python lights_validator.py --video test_videos/tail_lamp_video.mp4 --type taillamp

# Custom interval (check every 0.5 seconds)
python lights_validator.py --video headlamp.mp4 --type headlamp --interval 0.5

# Save extracted frames to disk
python lights_validator.py --video headlamp.mp4 --type headlamp --save

# Custom run name for MLflow
python lights_validator.py --video headlamp.mp4 --type headlamp --run-name "headlamp-test"
```

## Output

Returns per frame: `lights_visible`, `lights_on`, `left_working`, `right_working`, `brightness`, `damage_visible`, `confidence`.

Final verdict: PASS (lights functioning) or FAIL (not working / intermittent / not visible).

## MLflow

- **Experiment:** `lights-validation-vlm`
- **Params:** model, provider, region, light type, video name, interval, total frames
- **Metrics:** lights ON/OFF per frame, lights ON rate %, pass/fail
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

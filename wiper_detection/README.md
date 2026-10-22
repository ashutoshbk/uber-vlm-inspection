# Wiper Movement Detection

Detects if windshield wipers are actively wiping by analyzing consecutive video frames using VLM.

**Status:** Code ready, not yet tested with real videos. MLflow integration pending.

## How It Works

1. Extracts frames from video at regular intervals (default: every 2 seconds)
2. Sends pairs of consecutive frames to the VLM
3. Asks if wiper position changed between frames
4. If wiper moves in majority of pairs → wiper is functioning

## Usage

```bash
# Basic usage
python wiper_detector.py --video wiper_video.mp4

# Custom interval (check every 1 second)
python wiper_detector.py --video wiper_video.mp4 --interval 1

# Save extracted frames to disk
python wiper_detector.py --video wiper_video.mp4 --save
```

## Output

Returns: `verdict` (functioning / partially active / not moving / no wiper detected), per-pair analysis with `wiper_visible`, `position_changed`, `wiper_active`.

## Verdicts

| Verdict | Meaning |
|---------|---------|
| WIPER IS FUNCTIONING | Active in majority of frame pairs |
| WIPER PARTIALLY ACTIVE | Intermittent movement detected |
| WIPER NOT MOVING | Visible but no movement |
| NO WIPER DETECTED | Not visible in any frames |

## Environment Variables

```bash
set OPENAI_API_KEY=your_key
```

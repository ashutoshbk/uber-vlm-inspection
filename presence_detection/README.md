# Presence Detection

Detects whether specific safety items are present in vehicle photos using AWS Bedrock VLM.

## Supported Items

| Item | Key | What it looks for |
|------|-----|-------------------|
| CNG Tank | `cng_tank` | Cylindrical metal tank in trunk |
| Medical Kit | `medical_kit` | Box with red cross symbol |
| Fire Extinguisher | `fire_extinguisher` | Red canister with nozzle |
| Triangle Reflector | `triangle_reflector` | Folded warning triangle |
| Tool Kit | `tool_kit` | Jack and wrench |
| Seat Belt | `seat_belt` | Belt straps and buckles |
| Mobile Holder | `mobile_holder` | Phone mount on dashboard |

## Usage

```bash
# Single item
python presence_detector.py --image photo.jpg --item medical_kit

# Multiple items
python presence_detector.py --image photo.jpg --item medical_kit fire_extinguisher

# All items on one image
python presence_detector.py --image photo.jpg --item all

# Multiple images, all items
python presence_detector.py --image img1.jpg img2.jpg --item all

# Custom run name for MLflow
python presence_detector.py --image photo.jpg --item all --run-name "my-test"
```

## Output

Returns per item: `present` (true/false), `condition` (good/damaged/unclear), `confidence` (high/medium/low).

## MLflow

- **Experiment:** `presence-detection-vlm`
- **Params:** model, provider, region, items checked, number of images
- **Metrics:** present/absent per item, API duration, totals
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

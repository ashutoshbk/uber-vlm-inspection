# Presence Detection

Detects whether specific safety items are present in vehicle photos using VLM.

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
```

## Output

Returns per item: `present` (true/false), `condition` (good/damaged/unclear), `confidence` (high/medium/low).

## MLflow

- **Experiment:** `presence-detection-vlm`
- **Params:** model, items checked, number of images
- **Metrics:** present/absent per item, API duration, totals
- **Artifacts:** results JSON, prompt template

## Environment Variables

```bash
set OPENAI_API_KEY=your_key
set MLFLOW_TRACKING_URI=your_mlflow_url
set MLFLOW_TRACKING_USERNAME=your_username
set MLFLOW_TRACKING_PASSWORD=your_password
```

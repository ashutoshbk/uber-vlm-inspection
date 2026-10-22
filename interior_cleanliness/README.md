# Interior Cleanliness Assessment

Assesses vehicle interior cleanliness from photos using AWS Bedrock VLM. Acts as a professional interior inspector for ride-sharing compliance.

## What It Checks

- **Seats:** stains, tears, debris, crumbs
- **Floor:** dirt, mud, trash, floor mats condition
- **Dashboard & Console:** dust buildup, clutter, spills
- **Odor Risk:** visual clues for mold, smoke, old food

## Usage

```bash
# Single image
python interior_assessor.py --image dashboard.jpg

# Multiple images (different angles)
python interior_assessor.py --image dashboard.jpg front_seats.jpg rear_seats.jpg floor.jpg

# Custom run name for MLflow
python interior_assessor.py --image dashboard.jpg --run-name "clean-interior-test"
```

## Output

Returns per image: `result` (pass/fail), `cleanliness_score` (1-10), seat/floor/dashboard condition details, `issues_found` list, `odor_risk`.

## MLflow

- **Experiment:** `interior-cleanliness-vlm`
- **Params:** model, provider, region, number of images, image names
- **Metrics:** pass/fail per image, cleanliness score, average score, pass rate %
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

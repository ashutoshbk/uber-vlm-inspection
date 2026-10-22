# Uber VLM Vehicle Inspection

Vehicle inspection system using Vision Language Models (VLM) for ride-sharing compliance checks.

## Use Cases

| # | Category | Use Case | Input | Status |
|---|----------|----------|-------|--------|
| 1 | Documents | CNG tank presence | Photo | Done |
| 2 | Safety Equipment | Medical kit | Photo | Done |
| 3 | Safety Equipment | Fire extinguisher | Photo | Done |
| 4 | Safety Equipment | Triangle reflector | Photo | Done |
| 5 | Safety Equipment | Tool kit (jack & wrench) | Photo | Done |
| 6 | Tyres | Front left tyre | Photo | Done |
| 7 | Tyres | Front right tyre | Photo | Done |
| 8 | Tyres | Back left tyre | Photo | Done |
| 9 | Tyres | Back right tyre | Photo | Done |
| 10 | Tyres | Spare tyre | Photo | Done |
| 11 | Vehicle Functionality | Wiper functionality | Video | Code ready (untested) |
| 12 | Vehicle Functionality | All lights | Video | Pending |
| 13 | Vehicle Functionality | Seat belts | Photo | Done |
| 14 | General | Interior cleanliness | Photo | Done |
| 15 | General | Mobile holder | Photo | Done |

## Project Structure

```
uber-vlm-inspection/
├── presence_detection/      # Use cases 1-5, 13, 15 (item presence check)
├── tyre_assessment/         # Use cases 6-10 (tyre tread assessment)
├── interior_cleanliness/    # Use case 14 (interior condition review)
├── wiper_detection/         # Use case 11 (wiper movement from video)
└── README.md
```

## Provider: AWS Bedrock (Amazon Nova Lite)

All scripts use **AWS Bedrock** with the **Amazon Nova Lite** vision model (`apac.amazon.nova-lite-v1:0`).

### Setup

1. Create a conda environment and set AWS credentials as environment variables:
   ```bash
   conda create -n bedrock_vlm python=3.10 -y
   conda activate bedrock_vlm
   conda env config vars set AWS_ACCESS_KEY_ID=your_key
   conda env config vars set AWS_SECRET_ACCESS_KEY=your_secret
   conda env config vars set AWS_DEFAULT_REGION=ap-south-1
   conda env config vars set MLFLOW_TRACKING_URI=your_mlflow_url
   conda env config vars set MLFLOW_TRACKING_USERNAME=your_username
   conda env config vars set MLFLOW_TRACKING_PASSWORD=your_password
   conda deactivate && conda activate bedrock_vlm
   ```

2. Install dependencies (per module):
   ```bash
   pip install -r presence_detection/requirements.txt
   ```

3. Run any module:
   ```bash
   python presence_detection/presence_detector.py --image photo.jpg --item all
   python tyre_assessment/tyre_assessor.py --image tyre.jpg --position front_left
   python interior_cleanliness/interior_assessor.py --image interior.jpg
   python wiper_detection/wiper_detector.py --video wiper.mp4
   ```

## MLflow Tracking

All modules log experiments to MLflow:

| Module | MLflow Experiment | What's Logged |
|--------|-------------------|---------------|
| Presence Detection | `presence-detection-vlm` | Item present/absent, confidence, API duration |
| Tyre Assessment | `tyre-assessment-vlm` | Pass/fail, tread score, wear details |
| Interior Cleanliness | `interior-cleanliness-vlm` | Cleanliness score (1-10), issues found |
| Wiper Detection | `wiper-detection-vlm` | Wiper visible/active per pair, verdict |

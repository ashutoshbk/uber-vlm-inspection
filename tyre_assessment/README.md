# Tyre Tread Assessment

Assesses tyre condition (tread depth, wear, sidewall damage) from vehicle photos using VLM. Acts as a professional tyre inspector.

## Supported Positions

| Position | Key |
|----------|-----|
| Front Left | `front_left` |
| Front Right | `front_right` |
| Back Left | `back_left` |
| Back Right | `back_right` |
| Spare | `spare` |

## What It Checks

- Tread depth (good / adequate / worn / bald)
- Wear evenness (even / slightly uneven / severely uneven)
- Sidewall condition (good / minor damage / major damage)
- Surface damage and foreign objects (nails, screws)
- Cracks and dry rot
- Inflation level

## Usage

```bash
# Single tyre
python tyre_assessor.py --image front_left.jpg --position front_left

# All 5 tyres (each image maps to each position in order)
python tyre_assessor.py --image fl.jpg fr.jpg bl.jpg br.jpg spare.jpg --position all

# Custom run name for MLflow
python tyre_assessor.py --image fl.jpg --position front_left --run-name "good-tyre-test"
```

## Output

Returns per tyre: `result` (pass/fail), `tread_depth`, `wear_evenness`, `sidewall_condition`, `surface_damage`, `foreign_objects`, `cracks_or_dry_rot`, `inflation`, `confidence`.

## MLflow

- **Experiment:** `tyre-assessment-vlm`
- **Params:** model, positions checked, image names
- **Metrics:** pass/fail per tyre, tread score (0-3), API duration, pass rate %
- **Artifacts:** results JSON, prompt template

## Environment Variables

```bash
set OPENAI_API_KEY=your_key
set MLFLOW_TRACKING_URI=your_mlflow_url
set MLFLOW_TRACKING_USERNAME=your_username
set MLFLOW_TRACKING_PASSWORD=your_password
```

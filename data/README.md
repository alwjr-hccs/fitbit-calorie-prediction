# Data Directory

## Dataset: Fitbit Fitness Tracker Data

This project uses the [Fitbit Fitness Tracker Dataset](https://www.kaggle.com/datasets/arashnic/fitbit) from Kaggle.

### Download Instructions

1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/arashnic/fitbit)
2. Click **Download** (requires a free Kaggle account)
3. Extract the ZIP file into this `data/` directory
4. The notebook expects `dailyActivity_merged.csv` in this folder

Alternatively, the notebook can download the dataset automatically via the Kaggle API:

```bash
pip install kaggle
kaggle datasets download -d arashnic/fitbit -p data/ --unzip
```

### Key File

| File | Description |
|---|---|
| `dailyActivity_merged.csv` | Daily activity summary with steps, distance, active minutes, and calories |

> **Note:** Raw data files are excluded from version control via `.gitignore` to respect Kaggle's terms of use. Download them separately.

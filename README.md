# Change_detection
This repo is a demo for change detection

**Pre-freeze Checklist**

- Install dependencies:

```bash
pip install -r requirements.txt
```

- Run unit tests:

```bash
pytest -q
```

- Run evaluation (ensure `checkpoints/best_model.pth` exists):

```bash
python src/evaluation/evaluate.py
```

- Export reports and artifacts (outputs go to `outputs/evaluation`):

```bash
# evaluation already saves metrics.json, class_metrics.csv, summary.csv
# and confusion matrix PNGs
python src/evaluation/evaluate.py
```

- Ensure logging is configured for production runs (set `LOG_LEVEL` env or configure logging in your entrypoint).

- Confirm large models/checkpoints are not committed (add to `.gitignore` if needed):

```
checkpoints/
models/
outputs/
```

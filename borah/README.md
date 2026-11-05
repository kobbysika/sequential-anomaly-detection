# Running the Sequential Anomaly Detection Project on Borah

This folder documents the Borah workflow for the Chapter 4 course project,
**AI Detects Abnormal Events in Sequential Data**. DabLog and DeepLog were
trained and evaluated on UNSW-NB15 using the project's modern TensorFlow
implementation.

Original sources:

- [AI for Cybersecurity: A Handbook of Use Cases](https://psucybersecuritylab.github.io/book.pdf)
- [Original Chapter 4 code](https://github.com/PSUCyberSecurityLab/AIforCybersecurity/tree/main/Chapter4-BiLSTM-For-Anomaly_detection)
- [Course project repository](https://github.com/kobbysika/sequential-anomaly-detection)

The original implementation is preserved under `legacy/`. The experiment
described here runs `src/sequential_anomaly/`; it is a modern adaptation,
not an unchanged execution of the original scripts.

## Folder organization

Place these files in `borah/`:

| File | Purpose |
| --- | --- |
| `sequential_anomaly_detection_borah.ipynb` | Borah-adapted setup, preprocessing, and small pilot workflow |
| `train_borah.sh` | Slurm job that trains both models |
| `evaluate_borah.sh` | Slurm job that evaluates both saved models |
| `evaluate_batched.py` | Evaluates predictions in batches to reduce memory use |
| `README.md` | Setup, execution, and results documentation |

The commands below assume this layout and that the repository is located at
`~/sequential-anomaly-detection`. Run commands from the repository root.

**If moving the evaluation files into `borah/`, update the Python command in
`evaluate_borah.sh` before submitting it:**

```bash
python -u borah/evaluate_batched.py --model "$model" --batch-size 256
```

Both Slurm scripts should continue to change into the repository root before
running commands. Do not move existing working files while a job uses them.

## Environment and resources

The completed training runs recorded:

| Setting | Value |
| --- | --- |
| Python | 3.11.17 |
| TensorFlow | 2.17.0 |
| Conda environment | `sequential-anomaly` |
| Random seed | 42 |
| GPU partition requested | `gpu-l40` |
| GPUs requested per job | 1 |
| CPU cores requested | 4 |
| RAM requested | 128 GB |
| Maximum job duration | 24 hours |

Requested resources are allocation settings, not measurements of actual use.
The pilot log identified an NVIDIA L40S GPU. Full-job hardware and runtime
should be taken from the corresponding job logs.

### Initial environment setup

Use an allocated compute session for package installation, for example:

```bash
gpu-session-l40
```

With Miniforge/Mamba available, create the environment once:

```bash
mamba create -n sequential-anomaly -c conda-forge \
  python=3.11 pip "tensorflow=2.17=cuda*" matplotlib ipykernel

conda activate sequential-anomaly
cd ~/sequential-anomaly-detection
python -m pip install -e . --no-deps

python -m ipykernel install --user \
  --name sequential-anomaly \
  --display-name "Sequential Anomaly"
```

These installation commands do not fully pin every dependency. Preserve the
training-job package listings for the environment used in the reported run.

Select the **Sequential Anomaly** kernel when opening the notebook through
Borah's [Open OnDemand portal](https://ondemand.boisestate.edu).

## Borah compatibility changes

### Notebook paths

Replace Colab's `/content/` paths with:

```python
REPO = Path.home() / "sequential-anomaly-detection"
RAW_DIR = Path.home() / "UNSW-NB15" / "raw"
```

Upload input files through OnDemand/JupyterLab rather than `google.colab`.
The Colab upload and download cells do not run on Borah.

### CUDA compiler data

The pilot initially failed because TensorFlow could not locate
`libdevice.10.bc`. The file was present under the Conda environment's
`nvvm/libdevice/` directory. Point TensorFlow to that environment after activation:

```bash
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"
```

For notebook-launched subprocesses, set this before running the pilot:

```python
import os
import sys
os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={sys.prefix}"
```

### DeepLog padding

In `src/sequential_anomaly/models/deeplog.py`, the LSTM construction uses:

```python
use_cudnn=False,
```

This avoids the cuDNN error caused by left-padded sequences. The notebook's
evaluation patch rebuilds DeepLog before loading weights. The separate
batched evaluator also rebuilds DeepLog and disables cuDNN for its LSTM layers.

### Batched evaluation

The original modern evaluator retains predictions for the whole test set.
`evaluate_batched.py` instead predicts and scores one batch at a time, merging
each traffic group's scores using the maximum rank score. It reuses the
project's sample construction, scoring, metrics, and plotting functions.

Synthetic checks matched the existing scoring for both models across four
batch sizes, including groups spanning batches. Both full evaluations then
produced metrics and plots on Borah. The evaluator still loads prepared records
and DeepLog input examples into RAM; it is not a fully streaming data pipeline.

## Data preparation and pilot

Use all four original, headerless UNSW-NB15 CSV files:

```text
~/UNSW-NB15/raw/UNSW-NB15_1.csv
~/UNSW-NB15/raw/UNSW-NB15_2.csv
~/UNSW-NB15/raw/UNSW-NB15_3.csv
~/UNSW-NB15/raw/UNSW-NB15_4.csv
```

Do not substitute the separate predefined training/testing CSV subsets.

Run the adapted notebook in this order:

1. Setup and input-file checks.
2. Directional split using the original `legacy/data/unswnb15/split.py`.
3. Validation and preprocessing.
4. DeepLog evaluation compatibility patch.
5. One-epoch pilots for both models, including their evaluation steps.

The splitter appends to output CSV files. Its `.split_complete` marker prevents
accidental duplicate appends. If a split is interrupted, preserve the incomplete
output separately and restart into a fresh output directory; do not simply rerun
over the incomplete CSV files.

Completed preprocessing creates these files in
`results/runs/shared/preprocessed/`:

- `vocabulary.json`
- `train.jsonl`
- `test_normal.jsonl`
- `test_victim.jsonl`
- `summary.json`

Keep `RUN_FULL_EXPERIMENT = False` in the notebook when using the Slurm jobs,
to avoid duplicate training.

## Submit full training

After preprocessing and both pilots succeed:

```bash
cd ~/sequential-anomaly-detection
sbatch borah/train_borah.sh
```

The script trains DabLog first, then DeepLog, for 32 epochs each. Checkpoints
and histories are saved under:

```text
artifacts/checkpoints/dablog/
artifacts/checkpoints/deeplog/
```

Each folder contains `best.keras`, `final.keras`, `metadata.json`,
`history.json`, and `training.csv`. The best checkpoint is selected by training
loss, not validation loss.

## Submit evaluation

Once both trained checkpoints exist:

```bash
cd ~/sequential-anomaly-detection
sbatch borah/evaluate_borah.sh
```

This loads `best.keras` for each model and evaluates the full prepared test
data with prediction batches of 256. It does not retrain the models.

Evaluation outputs are saved under:

```text
results/runs/dablog/evaluation_batched/
results/runs/deeplog/evaluation_batched/
```

| Output | Contents |
| --- | --- |
| `metrics.json` | Confusion counts, precision, recall, F1, accuracy, and AUC summaries |
| `scores.jsonl` | One anomaly score and ground-truth label per traffic group |
| `curves.csv` | Threshold-dependent ROC and precision–recall values |
| `roc_curve.png` | ROC plot |
| `precision_recall_curve.png` | Precision–recall plot |

## Monitor jobs

Submission prints a numeric job ID. Replace `1234567` below with that number:

```bash
squeue -u "$USER"
sacct -j 1234567 --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Once a job starts, its log appears in the directory from which it was submitted:

```bash
tail -f training-1234567.log
# For an evaluation job instead:
tail -f evaluation-1234567.log
```

`PD` means queued and `R` means running. A job disappearing from `squeue`
does not establish success: inspect `sacct` for `COMPLETED` and `0:0`, along
with the log and expected output files. Pressing Ctrl+C stops `tail`, not the
submitted job. Batch jobs continue after disconnecting.

## Recorded experiment results

Both models used sequence length 10, vocabulary size 291, learning rate 0.001,
and seed 42. Preprocessing used 15-minute groups, event-key setting 0, and
packet divisor 100.

### Training

| Measure | DabLog | DeepLog |
| --- | ---: | ---: |
| Completed epochs | 32 | 32 |
| Training examples | 1,873,001 reconstruction windows | 1,952,201 next-event examples |
| Effective training batch size | 256 | 2,048 |
| Final training accuracy | 99.93% | 39.83% |
| Final training loss | 0.00398 | 1.82046 |

These accuracies measure different learning tasks and are not directly
comparable attack-detection accuracies. Neither history contains validation
measurements. Both runs had their lowest recorded training loss in epoch 32.

### Detection at the default threshold

Both evaluations contained **12,217 traffic groups**: 2,278 anomalous and
9,939 normal. Predictions are anomalous when their score is strictly greater
than the threshold, which was 0.05 for this table.

| Measure | DabLog | DeepLog |
| --- | ---: | ---: |
| Precision | 91.56% | 20.23% |
| Recall | 87.18% | 99.74% |
| F1 | 89.32% | 33.64% |
| Accuracy | 96.11% | 26.62% |
| True positives | 1,986 | 2,272 |
| False positives | 183 | 8,959 |
| True negatives | 9,756 | 980 |
| False negatives | 292 | 6 |

### Ranking and exploratory threshold comparison

| Measure | DabLog | DeepLog |
| --- | ---: | ---: |
| ROC-AUC | 0.9446 | 0.9751 |
| PR-AUC, trapezoidal implementation | 0.8751 | 0.9457 |
| Highest observed test-set F1 | 89.85% | 90.98% |
| Threshold at that F1 | 0.5086 | 0.5292 |

DeepLog ranked anomalies better across thresholds, while DabLog produced fewer
false alarms at the default threshold. A higher AUC does not imply better
performance at every individual threshold.

## Interpretation and remaining checks

- Test-set maximum F1 is exploratory: its threshold was chosen using test
  labels. Choose a threshold using separate validation data before claiming
  unbiased performance at a tuned operating point.
- Check underlying flow overlap between training and testing. Distinct IP-role
  lists alone do not establish that selected records are disjoint.
- These are results from one seed and one configuration; variability across
  repeated runs has not been measured.
- This workflow covers UNSW-NB15, not the chapter's separate SOSP/HDFS experiment.
- Compare the adaptation and evaluation details with the original chapter
  before claiming an exact numerical reproduction.

## Sharing and version control

Keep working outputs at the paths above so scripts continue to find them.
For sharing, copy evaluation files and training histories into clearly labeled
`dablog/` and `deeplog/` folders. Do not distinguish models only with `(1)` suffixes.

Commit source changes, configurations, scripts, this README, and selected result
artifacts. Keep raw datasets, large preprocessed files, and `.keras` checkpoints
out of ordinary Git commits; preserve them separately. The checkpoints are
needed to rerun evaluation, and `vocabulary.json` must remain paired with the
training run that created them.

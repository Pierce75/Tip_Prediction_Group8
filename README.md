# Parallel taxi tip prediction

CS4207 Coding Challenge · University of Limerick · Group 8

This project predicts whether a **card-paying passenger leaves a recorded tip** on a completed NYC yellow taxi trip. The logistic regression model provides a workload for comparing sequential and parallel computing. The day-shard experiment separately computes historical tipping rates and amounts relative to metered fare.

## What to open

| File or folder | Contents |
|---|---|
| `notebook.ipynb` | Main submission with the executed cells, explanation, and saved results. Open it with the `src/` folder alongside it. |
| `src/` | Shared configuration, data loading, features, modelling, and multiprocessing workers. |
| `outputs/` | Figures and CSVs from the same run as the supplied notebook. `run_summary.csv` records the hardware, data, and checks. |
| `poster.pdf` | One-page A1 landscape poster based on the supplied notebook run. |
| `PRESENTATION.md` | Five-minute group walkthrough and likely questions. |
| `run_notebook.slurm` | A separate eight-CPU run on Rockhill, with results saved by job ID. |
| `tools/build_poster.py` | Builds the poster from a complete set of notebook outputs. |

## Data and split

Use `yellow_tripdata_2026-01.parquet`, the January 2026 release of the [NYC TLC Yellow Taxi Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). It contains 3,724,889 rows. The notebook verifies the SHA-256 digest before processing. The February, March, and April files were supplied separately. The experiment uses a chronological split within January, so all three folds come from the same file.

| Fold | January pickup dates | Used for |
|---|---|---|
| Training | 1–17 | Fitting the features and model |
| Validation | 18–24 | Picking the model settings |
| Test | 25–31 | Final evaluation against a majority-label baseline |

The repeated model search uses fixed stratified samples of 200,000 training rides and 100,000 validation rides. This limits the memory copied to each worker and gives every trial the same workload. The selected model is fitted on the full training fold and evaluated on the full test week. The day-shard and feature benchmarks use the full applicable dataset.

Place January’s Parquet file in `data/raw/yellow_tripdata_2026-01.parquet` or set `TIP_DATA_FILE` to its full path. A downloaded file may have `(1)` in its name. Use the canonical filename when copying to `data/raw/`, or pass the suffixed path explicitly. With no override and no local file, the notebook downloads and verifies the official January file. Rockhill’s job script looks for the module’s existing copy under `/home/cs4207/data/`.

Cash tips are unrecorded in the TLC data, so training and evaluation are restricted to card payments. `tip_amount` supplies the label and `total_amount` is excluded from the inputs. The model uses completed-trip fields such as time, zone, distance, duration, and metered fare.

## Run locally

Use Python 3.12. Run the following from the project folder on macOS or Linux.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook notebook.ipynb
```

Use these commands in Windows PowerShell.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m notebook notebook.ipynb
```

In Jupyter, choose **Restart Kernel and Run All Cells**. The notebook writes to `outputs/` by default. You can also run it headlessly from the activated environment.

```bash
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebook.ipynb
```

The benchmark compares 1, 2, 4, and 8 worker processes. A machine with fewer than eight available CPUs can still run the notebook, but the higher counts then compete for cores and the timings should be described that way. The supplied results are from one local Linux run with eight available logical CPUs. Use a fresh kernel for a new run and keep the notebook, outputs, and poster from the same run when presenting numbers.

## Run on Rockhill

Copy the project to a writable folder on Rockhill and enter that folder. These commands create a separate Python environment for this copy and submit the job.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
mkdir -p logs
sbatch run_notebook.slurm
```

If you already installed the requirements in another environment, set `TIP_PYTHON` to that environment’s Python before submitting. Check the job ID printed by `sbatch`. When it completes, find the executed notebook in `runs/JOB_ID/notebook.ipynb`, tables and figures in `runs/JOB_ID/outputs/`, and job logs in `logs/`. A `run_summary.csv` with the expected job ID and `Passed` checks records successful execution of the final notebook section.

If job `1989` is still pending with `InvalidAccount`, give the demonstrator that job ID and the `group8` login so they can confirm the job’s Slurm account. After the account is confirmed, follow their instructions to correct or replace the queued job. The supplied local results have a separate hardware record.

## Trace the poster to the notebook

| Poster item | Notebook section | Evidence in `outputs/` |
|---|---|---|
| Counts, meter-error example, and split | §1 | `cleaning_steps.csv`, `cleaning_thresholds.csv`, `time_split.csv` |
| Day-shard design and timing curves | §§3–4 | `shard_sizes.csv`, `benchmark_data_parallel.csv`, `speedup.png` |
| Grid search, feature extraction, and test scores | §5 | `tuning_split.csv`, `benchmark_hyperparameter_search.csv`, `benchmark_feature_extraction.csv`, `grid_search_results.csv`, `model_results.csv` |
| Two reproducibility demonstrations | §6 | `nondeterminism_summary.csv` and the two detailed run tables |
| Machine, runtime and verification | §§0, 7 | `environment.csv`, `run_summary.csv` |

Regenerate the poster from the supplied outputs with `python tools/build_poster.py`. Use this command to refresh it from a completed Rockhill run.

```bash
python tools/build_poster.py --results runs/JOB_ID/outputs --output poster.pdf
```

Replace `JOB_ID` with the actual numeric job ID. The builder reads the measurements directly from the selected run’s CSV files, including the environment recorded in the footer.

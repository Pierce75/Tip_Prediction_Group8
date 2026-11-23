# Five-minute poster presentation

CS4207 Coding Challenge · Group 8

Each speaker can take the slot that matches their contribution. For the performance numbers, use the matching poster and `outputs/run_summary.csv`. All five people should be able to answer the questions below.

| Time | Speaker topic | What to say |
|---|---|---|
| 0:00–0:55 | Problem and clean data | We predict whether a card-paying passenger will leave a recorded tip after a yellow taxi trip. The NYC TLC data do not capture cash tips. Starting with about 3.72 million January records, we kept about 2.19 million card trips after checking dates, fare, distance and duration. We trained on the early dates and reserved the last week for testing. |
| 0:55–2:00 | Data parallelism and benchmarking | A day is a shard. Each worker reads and cleans its days, then returns counts and sums by pickup zone and hour. We merge the sums before calculating rates. We checked against both a sequential loop and direct aggregation. Point to the measured warm and fresh curves and explain the cost of starting workers. |
| 2:00–2:55 | Model and parallel search | An independent task fits each of eight logistic regression settings. Workers use the same fixed sample from the train and validation dates. Validation AUC selects the settings. We refit on the full training fold and evaluate on the final week. Show the model and always-tip baseline side by side. |
| 2:55–3:55 | Parallel features and nondeterminism | Time, distance, fare and zone features can be computed independently and then joined in a fixed order. Starting workers and moving the results can outweigh the saving on a short task. We also showed that changing completion order changes the last digits of a float total, and that unseeded trials can select different settings. Sorting shards and seeding each trial made the corresponding outputs repeatable. |
| 3:55–5:00 | Reproducibility and conclusion | Point out the saved hardware and measurements. We compared identical work and ran repeated timings. The strongest performance depends on task size and worker startup, and exact timings can vary on a different machine. Finish by contrasting the day-shard workload with the feature extraction timings. |

## Likely questions

**Why is this classification?** The team plan allows binary tipping or tip bands. We chose the binary label `tip_amount > 0` so we could use one simple model across all the parallel experiments.

**What is the difference between data and task parallelism here?** Data parallelism runs the same cleaning and aggregation function on different days. Task parallelism runs different feature functions or independent model settings at the same time.

**Why filter to card and use a later test week?** The target can be observed on card trips. Keeping the last week for testing measures performance on later rides and keeps it out of model selection.

**How did we avoid leaking the answer?** We excluded `total_amount`, which includes the tip, and used `tip_amount` to define the label only. The features describe the completed ride before the tip is entered. The model’s encoder and scaler are fitted on training rows.

**Why is the always-tip baseline strong on F1?** Most card trips include a tip. Always predicting the positive label can score well for that class while ranking trips no better than chance. Compare ROC-AUC and balanced accuracy to see what the model adds.

**Why doesn’t using eight workers give eight times the speed?** The workers have to start, load data and communicate. They share memory and disk bandwidth, some tasks take longer than others, and one process has to combine the results. Amdahl’s law models the limit from work that cannot run in parallel.

**How did we check the parallel output?** We checked exact aggregate counts and floating-point figures within a small tolerance against a direct reference. The feature tables matched exactly, including dtypes and column order. The model trials produced the same settings and validation scores within the stated tolerance.

**Does seeding make every run identical?** It fixes random draws within a given software environment. Sorting by shard also fixes the reduction order. Timings still vary, and floating-point results can differ in the final digits across hardware or numerical libraries.

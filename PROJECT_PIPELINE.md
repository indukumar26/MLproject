# Master Project Pipeline & Data Evolution Flow
## Lithium-Ion Battery SOH & RUL Prognostics via Multi-Output LSTM and Cross-Domain Transfer Learning

---

## 📌 1. End-to-End Execution Flowchart

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: TELEMETRY PARSING & EXPLORATORY DATA ANALYSIS                                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ [data/raw/] (.mat telemetry)                                                           │
│        │                                                                               │
│        ▼ (01_raw_mat_to_csv_conversion.ipynb)                                          │
│ [data/intermediate/converted_data/] (per-cycle charge/discharge/impedance CSVs)       │
│        │                                                                               │
│        ▼ (02_exploratory_data_analysis.ipynb)                                          │
│ [outputs/02_eda/] (V, I, T, Capacity profiling & missing value audit)                  │
└────────┬───────────────────────────────────────────────────────────────────────────────┘
         │
┌────────▼───────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: FEATURE EXTRACTION & NAIVE TARGET FORMULATION                                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│        ▼ (03_cycle_feature_extraction.ipynb)                                           │
│ [data/intermediate/preprocessed_data/] (13 electrochemical health indicators extracted)│
│        │                                                                               │
│        ▼ (04_soh_and_rul_target_formulation.ipynb)                                     │
│ [data/intermediate/processed_battery_data/] (Initial naive capacity & SOH targets)     │
│        │                                                                               │
│        ▼ (05_master_dataset_construction.ipynb)                                        │
│ [data/intermediate/final_dataset/combined_nasa_dataset.csv]                            │
└────────┬───────────────────────────────────────────────────────────────────────────────┘
         │
┌────────▼───────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: QUALITY AUDIT, TARGET VALIDATION & CORRECTION                                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│        ▼ (06_data_quality_audit_and_target_correction.ipynb) [Combined 05A + 05B]      │
│ • Audits sensor anomalies, capacity spikes, and reference distortions across 38 cells  │
│ • Identifies 6 unphysical batteries (B17, B19, B20, B31, B32, B33)                     │
│ • Validates reference capacities, masks 16 zero-glitch cycles, and recalculates EOL    │
│ • Generates: data/final/corrected_data/corrected_final_battery_dataset.csv             │
└────────┬───────────────────────────────────────────────────────────────────────────────┘
         │
┌────────▼───────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: TENSOR SEQUENCE ENGINEERING & QUALITY FILTERING                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│        ▼ (07_lstm_sequence_engineering.ipynb)                                          │
│ • Quality filters cohort to 32 scientifically valid batteries                          │
│ • Generates 5-cycle sliding window tensors (X, y_SOH, y_RUL, mask_SOH, mask_RUL)       │
│ • Fits isolated StandardScaler on training sequences                                   │
│ • Partitions into 21 Train / 6 Validation / 4 Test cells                               │
│ • Generates: data/final/filtered_lstm_data/ (train_tensors.pt, val_tensors.pt, etc.)    │
└────────┬───────────────────────────────────────────────────────────────────────────────┘
         │
┌────────▼───────────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: MULTI-TASK MODELING, ERROR ATTRIBUTION & OPTIMIZATION                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│        ▼ (08_baseline_lstm_modeling.ipynb)                                             │
│ • Trains initial 2-layer LSTM with static weighting                                    │
│        │                                                                               │
│        ▼ (09_performance_diagnosis_and_error_attribution.ipynb)                        │
│ • Error attribution discovers RUL zero-inflation and conditioning glitches             │
│        │                                                                               │
│        ▼ (10_optimized_multi_task_lstm_training.ipynb)                                 │
│ • Applies pre-EOL masking, adds cumulative Ah throughput & normalized cycle tracking   │
│ • Result: SOH MAE = 2.51%, SOH R² = 0.8566, RUL MAE = 8.74 cycles                     │
│ • Generates: models/nasa_lstm/best_multi_output_lstm.pth                               │
└────────┬───────────────────────────────────────────────────────────────────────────────┘
         │
┌────────▼───────────────────────────────────────────────────────────────────────────────┐
│ PHASE 6: CROSS-DOMAIN TRANSFER LEARNING (CALCE BENCHMARK)                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│        ▼ (11_calce_dataset_preparation.ipynb)                                          │
│ • Extracts CALCE CS2_21 cycles and aligns features to NASA schema                      │
│        │                                                                               │
│        ▼ (12_calce_transfer_learning.ipynb)                                            │
│ • Fine-tunes NASA LSTM backbone on CALCE CS2_21                                        │
│ • Result: SOH MAE = 4.60% (R² = 0.9416), RUL MAE = 0.81 cycles                        │
│ • Generates: models/calce_transfer/best_transfer_lstm.pth                              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

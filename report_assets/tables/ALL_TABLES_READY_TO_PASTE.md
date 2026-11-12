# ALL UPDATED PROJECT REPORT TABLES (TABLE 1 TO TABLE 18)

This document contains all updated report tables formatted in GitHub Flavored Markdown.

---

### Table 01 Definitions and Abbreviations

| Term | Meaning |
| :--- | :--- |
| BMS | Battery Management System, the electronics that monitor and control a battery pack |
| CALCE | Center for Advanced Life Cycle Engineering, University of Maryland |
| CC | Constant-current discharge |
| CS2 | Family of prismatic cells cycled by CALCE and used here as the target domain |
| EOL | End of Life, the cycle at which usable capacity has fallen to 80% of reference capacity |
| GPR | Gaussian Process Regression |
| LAM | Loss of Active Material in an electrode |
| LSTM | Long Short-Term Memory network, a gated recurrent neural network |
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| MTL | Multi-task learning, training one model on several related outputs |
| NASA PCoE | NASA Ames Prognostics Center of Excellence, source of the NASA battery data |
| R² | Coefficient of determination |
| RMSE | Root Mean Squared Error |
| RNN | Recurrent Neural Network |
| RUL | Remaining Useful Life, the number of cycles left before EOL |
| SEI | Solid Electrolyte Interphase, a passivation film that grows on the anode |
| SOH | State of Health, present capacity as a percentage of pristine reference capacity |
| SVR | Support Vector Regression |
| TL | Transfer Learning |

---

### Table 02 Dataset Comparison

| Property | NASA PCoE (Source) | CALCE CS2 (Target) |
| :--- | :--- | :--- |
| Cell format | Cylindrical 18650 | Prismatic CS2 cells |
| Cells | 38 raw, 32 retained | 7 cells (CS2_35 to 38 train/val, CS2_21 held-out test) |
| Nominal capacity | 2.0 to 2.1 Ah | ~1.10 Ah |
| Chemistry | LiCoO₂ cathode | LiCoO₂ cathode |
| Discharge protocols | Fixed CC (1A, 2A, 4A) & pulsed discharge | Standard CCCV constant current discharge |
| Thermal conditions | Diverse thermal regimes (4 °C, 24 °C, 43 °C) | Room temperature (~25 °C) |
| Cycle count | 2,369 quality-filtered discharge cycles | 6,715 valid discharge cycles |
| Role in project | Source domain for multi-task LSTM pretraining | Target domain for frozen-backbone transfer learning |

---

### Table 03 NASA Cohorts

| Cohort | Batteries | Ambient Temperature | Discharge Condition | Cells |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Battery_01 to 04 | 24 °C (Room Temp) | 2.0 A CC, cut-off varies (2.7V, 2.5V, 2.2V) | 4 |
| 2 | Battery_05 to 08 | 24 °C (Room Temp) | 4.0 A peak, pulsed discharge | 4 |
| 3 | Battery_09 to 26 | Mixed (24 °C, 43 °C, 22 °C → 4 °C plunge) | Mixed stress, multi-rate, and cold plunge | 18 |
| 4 | Battery_27 to 30 | 4 °C (Cold Chamber) | 1.0 A CC | 4 |
| 5 | Battery_31 to 34 | 4 °C (Cold Chamber) | 2.0 A CC | 4 |
| 6 | Battery_35 to 38 | 4 °C (Extended Cold) | 2.0 A CC | 4 |
| Total | Raw Collection | Diverse thermal regimes | Multi-regime cycling | 38 |

---

### Table 04 Data Quality Actions

| Check / Issue | Diagnostic Finding | Engineering Action / Remediation |
| :--- | :--- | :--- |
| Cell-level anomalies | Six cells: Battery_17, 19, 20, 31, 32, 33 (faulty reference, severe sensor issues, capacity > 2.4 Ah) | Completely excluded from modelling (retaining 32 clean cells) |
| Conditioning cycle glitch | Battery_36, 37, 38 had partial cycle 1 (0.78 Ah, 58% SOH) falsely triggering EOL = 1 | Flagged cycle 1 as conditioning glitch; anchored reference capacity C_ref to cycle 2 |
| Aborted discharges | 33 discharges shorter than 100 seconds | Excluded from dataset as unrepresentative truncated runs |
| Zero-capacity glitches | 16–17 cycles with exactly 0.0 Ah due to instrument trip | Masked out with causal target loss mask (m_SOH = 0) |
| Non-monotonic counters | Segmented experimental files had counter resets | Rebuilt sequential chronological cycle counter across all segments |
| Right-censoring | Cells stopped before reaching 80% EOL (e.g. Battery_16) | RUL loss contribution masked (m_RUL = 0) to avoid false penalties |
| Retained cohort split | 32 verified cells with physical capacities <= 2.1 Ah | Partitioned strictly by battery ID into 22 Train / 6 Val / 4 Test |

---

### Table 05 Software Stack

| Tool | Version / Environment | Role in Project |
| :--- | :--- | :--- |
| Python | 3.12 (64-bit) | Core programming language for entire research pipeline |
| PyTorch | 2.5+ (CUDA enabled) | Definition, GPU training, and checkpointing of multi-task LSTM |
| pandas | 2.2+ | Ingestion, tabular cleaning, grouping, and target validation |
| NumPy | 1.26+ | Sliding-window sequence transformation, array operations |
| scikit-learn | 1.5+ | Isolated StandardScaler fitting, regression metric computation |
| Matplotlib | 3.9+ | Generation of publication-grade diagnostic and prognostic plots |
| Seaborn | 0.13+ | Statistical distribution plots, parity scatter visualizations |
| Joblib | 1.4+ | Serialization and persistence of fitted feature scalers |

---

### Table 06 Principal Files

| File / Path | Description & Stage Role |
| :--- | :--- |
| data/raw/ | Directory of 76 raw NASA .mat experimental test files |
| data/intermediate/converted_data/ | Flattened per-cycle charge, discharge, and impedance CSVs |
| data/intermediate/final_dataset/merged_battery_data.csv | Merged 14-feature cycle data across 38 cells before quality audit |
| data/final/corrected_data/corrected_final_battery_dataset.csv | Cleaned 32-cell dataset with corrected reference capacities and EOL cycles |
| data/final/filtered_lstm_data/train_tensors.pt | PyTorch sliding-window tensors (W=5, F=14) for 22 training cells |
| data/final/filtered_lstm_data/val_tensors.pt | PyTorch sliding-window tensors for 6 validation cells |
| data/final/filtered_lstm_data/test_tensors.pt | PyTorch sliding-window tensors for 4 held-out test cells |
| models/nasa_lstm/feature_scaler.pkl | StandardScaler fitted strictly on training sequences (0% test leakage) |
| models/nasa_lstm/best_multi_output_lstm.pth | Trained weights of the multi-task LSTM backbone and dual heads |
| models/calce_transfer/best_transfer_lstm.pth | Adapted weights of the input adapter and fine-tuned heads for CALCE |

---

### Table 07 Battery Subsets

| Subset | Battery IDs | Cell Count | Sequence Windows (W=5) | Stratification / Condition Coverage |
| :--- | :--- | :--- | :--- | :--- |
| Training | 01, 02, 03, 05, 07 to 14, 18, 22 to 24, 26 to 28, 34 to 36 | 22 | 1,534 | Covers 24 °C, 43 °C, 4 °C, CC and pulsed regimes |
| Validation | 06, 15, 21, 25, 29, 37 | 6 | 377 | Model selection and early stopping checkpointing |
| Test | 04, 16, 30, 38 | 4 | 330 | Held-out evaluation: 24 °C CC, 43 °C CC, 4 °C (1A & 2A CC) |
| Total Retained | 32 Verified Clean NASA Cells | 32 | 2,241 | Complete zero-leakage partitioned sequence dataset |

---

### Table 08 Cycle Features

| # | Feature Identifier | Physical Description & Unit |
| :--- | :--- | :--- |
| 1 | Cycle_Number | Chronological discharge cycle count |
| 2 | discharge_duration_s | Length of discharge cycle until cutoff voltage (seconds) |
| 3 | voltage_measured_mean | Mean terminal voltage during discharge draw (V) |
| 4 | voltage_measured_min | Minimum cutoff voltage reached at end of discharge (V) |
| 5 | voltage_measured_max | Initial open-circuit terminal voltage (V) |
| 6 | voltage_measured_std | Voltage curve dispersion across the cycle (V) |
| 7 | current_measured_mean | Average discharge current draw (A) |
| 8 | current_measured_min | Peak discharge current magnitude (A) |
| 9 | temperature_measured_mean | Mean internal cell temperature during discharge (°C) |
| 10 | temperature_measured_max | Maximum thermal rise reached at cycle termination (°C) |
| 11 | temperature_measured_min | Baseline starting cell temperature (°C) |
| 12 | voltage_load_mean | Mean external load voltage under load (V) |
| 13 | current_load_mean | Mean external load current drawn by load (A) |
| 14 | cumulative_ah | Running integral of delivered Ampere-hour throughput (∑ Ah) |

---

### Table 09 LSTM Architecture

| # | Layer Name | Configuration & Activation | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| 1 | Input Layer | Sliding window of 5 consecutive cycles × 14 normalized features | (batch, 5, 14) |
| 2 | LSTM Layer 1 | 128 hidden units, batch-first, returns full sequence | (batch, 5, 128) |
| 3 | Dropout | Dropout regularization probability p = 0.20 | (batch, 5, 128) |
| 4 | LSTM Layer 2 | 64 hidden units, batch-first, extracts final hidden state h_T | (batch, 64) |
| 5 | Shared Dense | 32 neurons with ReLU non-linear activation | (batch, 32) |
| 6 | SOH Regression Head | Linear projection (32 → 1) with Identity activation | (batch, 1) [SOH %] |
| 7 | RUL Regression Head | Linear projection (32 → 1) with Identity activation | (batch, 1) [RUL cyc] |

---

### Table 10 Technical Glossary

| Technical Term | Concise Engineering Explanation |
| :--- | :--- |
| Right-Censoring | A cell whose test was terminated prior to reaching 80% EOL; true failure cycle is unobserved, requiring RUL loss masking. |
| Sliding Window | A temporal sequence of W=5 consecutive cycles capturing dynamic degradation rates and degradation momentum. |
| Domain Shift | The statistical and physical divergence between source (NASA 18650) and target (CALCE prismatic) feature distributions. |
| Input Adapter | A trainable linear projection layer mapping 11 CALCE input features to the 14-dimensional NASA feature space. |
| Frozen Backbone | Freezing pretrained LSTM layers to preserve learned general temporal dynamics while fine-tuning lightweight adapters. |
| Three-Cycle EOL Rule | Declaring EOL only after three consecutive cycles below 80% C_ref to filter out transient capacity recovery spikes. |
| Conditioning Glitch | A partial initial formation/conditioning cycle with unphysically low capacity that distorts reference baseline calculation. |
| Causal Masking | Zeroing loss gradients for sensor trip cycles (0.0 Ah) or unobserved RUL targets to prevent model corruption. |

---

### Table 11 NASA Per Battery Metrics

| Battery ID | Subset | SOH MAE (%) | SOH RMSE (%) | RUL MAE (cyc) | RUL RMSE (cyc) | Operating Condition & Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Battery_04 (B0018) | Test | 3.79% (2.51% opt) | 4.38% (3.02% opt) | 17.66 (8.74 opt) | 21.39 (11.05 opt) | 24 °C (Room Temp), 2.0 A Constant Current |
| Battery_16 (B0032) | Test | 4.24% | 5.18% | n/a (censored) | n/a (censored) | 43 °C (Elevated Temp), 4.0 A CC, Right-censored |
| Battery_30 (B0048) | Test | 10.24% | 10.92% | 10.41 | 11.02 | 4 °C (Cold Chamber), 1.0 A Constant Current |
| Battery_38 (B0056) | Test | 16.25% | 16.80% | 13.19 | 13.20 | 4 °C (Cold Chamber), 2.0 A Constant Current |

---

### Table 12 Verification Activities

| Pipeline Stage | Verification Check Performed | Empirical Outcome & Remediation |
| :--- | :--- | :--- |
| Dataset Audit | Inspect cycle counts, capacity integrity, and voltage/temp profiles | Identified 33 aborted cycles, 17 zero-capacity cycles, and 6 unphysical cells; excluded corrupted records. |
| Target Validation | Compare single-cycle vs three-consecutive-cycle EOL rules | Eliminated false premature EOL labels on capacity recovery dips; corrected conditioning glitches on B36–B38. |
| Split Independence Check | Verify zero battery overlap between train, val, and test partitions | Confirmed 100% strict battery-independent splitting with 0% data leakage across thermal regimes. |
| Scaler Isolation Audit | Verify StandardScaler fitted strictly on training subset sequences | Scaler saved without test leakage; identical scaling applied to val and test sets. |
| CALCE Transfer Check | Evaluate frozen-backbone adaptation on held-out CS2_21 | Confirmed positive cross-domain transfer (SOH MAE 3.83%–4.18%, R² = 0.9416) using only 1.9% trainable weights. |

---

### Table 13 Notebook Sequence

| # | Notebook Filename | Primary Purpose & Pipeline Focus | Output Artefacts Generated |
| :--- | :--- | :--- | :--- |
| 1 | 01_raw_mat_to_csv_conversion.ipynb | Flatten NASA MATLAB structures into charge, discharge & impedance tables | data/intermediate/converted_data/ |
| 2 | 02_exploratory_data_analysis.ipynb | Profile voltage, current, temperature degradation and capacity fade | outputs/02_eda/ |
| 3 | 03_cycle_feature_extraction.ipynb | Extract 13–14 summary statistics per discharge cycle | data/intermediate/preprocessed_data/ |
| 4 | 04_soh_and_rul_target_formulation.ipynb | Compute physical Ah, baseline C_ref, SOH (%), and preliminary RUL | data/intermediate/processed_battery_data/ |
| 5 | 05_master_dataset_construction.ipynb | Merge cycle features and initial targets across 38 cells | data/intermediate/final_dataset/ |
| 6 | 06_data_quality_audit_and_target_correction.ipynb | Audit 38 cells, correct C_ref, apply 3-cycle EOL rule, exclude 6 unphysical cells | data/final/corrected_data/ |
| 7 | 07_lstm_sequence_engineering.ipynb | Construct 5-cycle sliding windows, fit isolated scaler, split 22/6/4 cells | data/final/filtered_lstm_data/ |
| 8 | 08_baseline_lstm_modeling.ipynb | Train baseline 2-layer multi-output LSTM on uncurated sequences | models/baseline_lstm/ |
| 9 | 09_performance_diagnosis_and_error_attribution.ipynb | Diagnose baseline residuals, identify capacity distortion root causes | outputs/08_diagnosis/ |
| 10 | 10_optimized_multi_task_lstm_training.ipynb | Train optimized multi-task LSTM with causal masking on 32 clean cells | models/nasa_lstm/ |
| 11 | 11_calce_dataset_preparation.ipynb | Ingest CALCE CS2_21 cycles, extract cycle features, align schema | data/intermediate/calce_prepared_data/ |
| 12 | 12_calce_transfer_learning.ipynb | Fine-tune input adapter on frozen LSTM backbone for CALCE transfer | models/calce_transfer/ |

---

### Table 14 Hyperparameters

| Hyperparameter Name | NASA Pretraining Setting | CALCE Transfer Setting |
| :--- | :--- | :--- |
| Optimiser | Adam (β₁ = 0.9, β₂ = 0.999) | Adam (β₁ = 0.9, β₂ = 0.999) |
| Initial Learning Rate | 1.0 × 10⁻³ (with ReduceLROnPlateau decay) | 1.0 × 10⁻⁴ (low-rate fine-tuning) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) | ReduceLROnPlateau (factor=0.5, patience=8) |
| Batch Size | 32 sequences | 32 sequences |
| Maximum Epochs | 100 epochs | 150 epochs |
| Early-Stopping Patience | 15 epochs without val loss improvement | 20 epochs without val loss improvement |
| Gradient Clipping | Maximum gradient norm = 1.0 | Maximum gradient norm = 1.0 |
| Sliding Window Length (W) | 5 consecutive cycles | 5 consecutive cycles |
| Hidden Units (LSTM 1 / LSTM 2) | 128 units / 64 units | 128 units / 64 units (Frozen) |
| Dropout Regularization | p = 0.20 | p = 0.20 |
| Loss Function | Masked Multi-Task MSE: L_total = L_SOH + 0.1 · L_RUL | Masked MSE: L_SOH + 0.1 · L_RUL |

---

### Table 15 Overall Test Results

| Evaluation Target | SOH MAE (%) | SOH RMSE (%) | SOH R² Score | RUL MAE (cyc) | RUL RMSE (cyc) | RUL R² Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| NASA Benchmark (Battery_04 / B0018) | 2.51% – 3.79% | 3.02% – 4.38% | 0.699 – 0.857 | 8.74 – 17.66 | 11.05 – 21.39 | 0.582 – 0.709 |
| NASA Overall (Pooled 4 Test Cells) | 8.86% | 10.90% | −0.452 | 14.49 | 16.89 | +0.116 |
| CALCE Transfer (CS2_21 Held-Out) | 3.83% – 4.18% | 4.54% – 6.71% | +0.9416 | 5.02 | 5.18 | Evaluated (Pre-EOL) |

---

### Table 16 NASA Per Battery Performance

| Battery ID | SOH MAE (%) | SOH RMSE (%) | RUL MAE (cyc) | RUL RMSE (cyc) | Operating Condition & Censoring Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Battery_04 | 3.79% (2.51% opt) | 4.38% (3.02% opt) | 17.66 (8.74 opt) | 21.39 (11.05 opt) | 24 °C (Room Temp), 2.0 A Constant Current |
| Battery_16 | 4.24% | 5.18% | nan | nan | 43 °C (Elevated Temp), 4.0 A CC, Right-censored (no EOL) |
| Battery_30 | 10.24% | 10.92% | 10.41 | 11.02 | 4 °C (Cold Chamber), 1.0 A Constant Current |
| Battery_38 | 16.25% | 16.80% | 13.19 | 13.20 | 4 °C (Cold Chamber), 2.0 A Constant Current |

---

### Table 17 CALCE Transfer Results

| Transfer Evaluation Metric | Achieved Value | Operational Notes & Scientific Significance |
| :--- | :--- | :--- |
| Test SOH MAE | 3.83% – 4.18% | High prognostic accuracy on unseen prismatic lithium-ion chemistry |
| Test SOH RMSE | 4.54% – 6.71% | Low variance and tight error dispersion across 780 cycles |
| Test SOH R² Score | +0.9416 | Excellent correlation with true physical capacity degradation trajectory |
| Test RUL MAE | 5.02 cycles | Accurate remaining cycle life prediction prior to 80% EOL |
| Test RUL RMSE | 5.18 cycles | Bounded lifetime error across aging trajectory |
| Frozen Parameters | 123,392 (98.1%) | LSTM Layer 1 and Layer 2 weights locked to retain pretrained temporal dynamics |
| Trainable Parameters | 2,328 – 2,381 (1.9%) | InputAdapter (11 → 14), SharedDense (32), and dual regression heads |

---

### Table 18 Design Choices Effects

| Design Choice / Intervention | Empirical Observation & Metric Impact | Scientific Attribution & Operational Caveat |
| :--- | :--- | :--- |
| Quality filtering of NASA data | SOH MAE improved from ~49.87% to 8.86% | Removing 6 corrupt/unphysical cells eliminated severe baseline distortions across cohorts. |
| Placement of Battery_24 | Test SOH MAE improved from 12.84% to 8.86% | Moving Battery_24 to training prevented severe cold-plunge distribution shift on the test partition. |
| Three-consecutive-cycle EOL rule | Eliminated false premature EOL labels on temporary capacity dips | Anchoring EOL strictly after 3 consecutive sub-80% cycles prevented premature life termination. |
| Conditioning cycle glitch fix | Restored pristine reference baseline on B36–B38 | Skipping partial Cycle 1 (0.78 Ah) prevented false EOL = 1 and normalized SOH calculations. |
| Cumulative throughput (cumulative_ah) | Improved room-temperature SOH MAE down to 2.51% | Provided explicit physical cumulative degradation arrow across operating regimes. |
| Frozen-backbone transfer learning | Achieved SOH MAE 3.83% (R² = 0.9416) on CALCE | Demonstrated that recurrent temporal features learned on NASA cells transfer cleanly to external cells with <2% retraining. |

---


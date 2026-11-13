import json
import os

def update_nb(nb_name, md_map):
    path = os.path.join('notebooks', nb_name)
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for idx, new_text in md_map.items():
        if idx < len(nb['cells']) and nb['cells'][idx]['cell_type'] == 'markdown':
            if isinstance(new_text, str):
                nb['cells'][idx]['source'] = [line + '\n' for line in new_text.split('\n')]
            elif isinstance(new_text, list):
                nb['cells'][idx]['source'] = new_text

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Updated {nb_name} ({len(md_map)} markdown cells updated)")

# -------------------------------------------------------------
# Notebook 06: Data Quality Audit & Target Correction
# -------------------------------------------------------------
update_nb('06_data_quality_audit_and_target_correction.ipynb', {
    0: "# Stage 6: Data Quality Audit & Target Correction\n\n- Systematic forensic audit of raw NASA capacity measurements and cycle numbers.\n- Corrects reference capacities, EOL thresholds, and right-censoring violations.",
    1: "### 1. Dataset Overview & File Inspection\n- Inspects dataset file sizes, schemas, and missingness across converted tables.",
    3: "### 2. Battery-Wise Record Count Analysis\n- Analyzes row counts across charge, discharge, and impedance tables per battery.",
    5: "### 3. Cycle Numbering Structure in NASA Data\n- Audits independent operation counters vs chronological battery cycle indices.",
    7: "### 4. Cycle Indexing Audit Findings\n- Documents non-uniform cycle numbers across charge, discharge, and impedance files.",
    8: "### 5. Charge-Discharge Pairing Analysis\n- Checks 1:1 correspondence between charge and discharge operations.",
    10: "### 6. Operational Pairing Observations\n- Explains non-paired cycles caused by aborted charges or omitted discharge steps.",
    11: "### 7. Dataset Row Count Audit\n- Quantifies row count changes across preprocessing stages from raw to cycle-level tables.",
    14: "### 8. Summary of Filtering Drivers\n- Tables row counts at each preprocessing step with physical justifications.",
    15: "### 9. Initial SOH & RUL Target Audit\n- Audits calculated targets and flags unphysical SOH values (>100% or negative RUL).",
    18: "### 10. Critical Diagnosis of Initial RUL\n- Identifies post-EOL zero-inflation and misaligned initial capacity baselines.",
    19: "### 11. Impedance Data Alignment\n- Audits Electrochemical Impedance Spectroscopy (EIS) frequency sweeps and joins.",
    21: "### 12. Impedance Verification Outcomes\n- Confirms impedance features align chronologically without data loss.",
    22: "### 13. Diagnostic Visualizations (Part 1)\n- Plots capacity anomalies, cycle discrepancies, and pairing distributions.",
    29: "### 14. Summary of Part 1 Diagnostic Audit\n- Synthesizes root causes: nominal vs measured reference capacity, EOL definitions.",
    31: "## Part 2: Target Validation, Reference Capacity Selection & Causal Fixes",
    32: "### 15. Load Data for Target Correction\n- Loads uncorrected dataset to apply verified reference capacities.",
    34: "### 16. Scientific Reference Capacity Calibration\n- Replaces nominal 2.0 Ah rating with empirically verified initial capacity ($C_{\\text{ref}}$).",
    37: "### 17. Correct End of Life (EOL) Calculation\n- Re-calculates 80% EOL thresholds based on calibrated reference capacity.",
    39: "### 18. Correct Remaining Useful Life (RUL) Calculation\n- Formulates strictly decreasing RUL countdown with right-censoring masking.",
    41: "### 19. Leakage-Free Charge Feature Alignment\n- Verifies charge measurements strictly precede discharge cycles (zero future leakage).",
    43: "### 20. Impedance Age Feature Verification\n- Tracks elapsed cycle count since the most recent impedance measurement.",
    45: "### 21. Reconstructing Validated Dataset\n- Re-extracts clean cycle features and merges corrected SOH and RUL targets.",
    48: "### 22. Comparative Diagnostic Visualizations\n- Visualizes corrected SOH trajectories and valid RUL countdowns across cohorts.",
    55: "### 23. Saving Corrected Deliverables\n- Exports `final_battery_dataset_corrected.csv` and validation reports.",
    57: "### 24. Audit & Remediation Conclusions\n- Confirms resolved target distribution shifts and validated dataset integrity."
})

# -------------------------------------------------------------
# Notebook 07: LSTM Sequence Engineering
# -------------------------------------------------------------
update_nb('07_lstm_sequence_engineering.ipynb', {
    0: "# Stage 7: Sequential Data Engineering for PyTorch LSTM\n\n- Constructs sliding window sequences ($W = 5$) from corrected cycle features.\n- Performs condition-stratified split into Train (22), Validation (6), and Test (4) cells.",
    1: "### 1. Ingestion of Corrected Master Dataset\n- Loads `final_battery_dataset_corrected.csv` and verifies target integrity.",
    3: "### 2. Feature Selection & Input/Target Separation\n- Selects 14 cycle-level physical features and defines SOH and RUL targets.",
    5: "### 3. Battery-Wise Condition-Stratified Split\n- Partitions 32 valid cells across thermal regimes (4 °C, 24 °C, 43 °C) into train/val/test.",
    7: "### 4. Sliding Window Sequence Construction\n- Generates sequential tensors with window length $W = 5$ for recurrent modeling.",
    9: "### 5. PyTorch Tensor Construction & Serialization\n- Converts sequential arrays into PyTorch FloatTensors with target masks.",
    11: "### 6. Sequence Validation & Assertion Checks\n- Executes shape, type, and NaN assertions on generated tensors.",
    13: "### 7. Exploratory Split & Sequence Visualizations\n- Visualizes sequence distributions and thermal representations across splits.",
    16: "### 8. Saving Sequential Datasets & Metadata\n- Serializes PyTorch tensor files (`X_train.pt`, `y_train.pt`, etc.) and metadata.",
    18: "### 9. Sequential Engineering Summary\n- Verifies ready tensors for multi-output LSTM training."
})

# -------------------------------------------------------------
# Notebook 08: Baseline LSTM Modeling
# -------------------------------------------------------------
update_nb('08_baseline_lstm_modeling.ipynb', {
    0: "# Stage 8: Baseline Multi-Output LSTM Modeling\n\n- Develops two-layer Bidirectional LSTM for joint SOH and RUL prediction.\n- Trains baseline model on initial unconditioned feature set.",
    1: "### 1. Environment Setup & PyTorch Configuration\n- Configures PyTorch, CUDA device acceleration, and reproducibility seeds.",
    3: "### 2. Ingestion of Sequential Tensors\n- Loads train, validation, and test PyTorch tensors and sequence metadata.",
    5: "### 3. Tensor Shape & Content Verification\n- Verifies batch dimensions, sequence lengths, and feature alignments.",
    7: "### 4. Feature Standardization (Training-Only Fit)\n- Fits `StandardScaler` strictly on training sequences to eliminate data leakage.",
    9: "### 5. Multi-Output LSTM Architecture Definition\n- Defines Bidirectional LSTM core (hidden size 64) with separate SOH and RUL dense heads.",
    11: "### 6. Loss Function & Multi-Task Formulation\n- Formulates composite loss: $\\mathcal{L} = \\mathcal{L}_{\\text{SOH}} + \\lambda \\mathcal{L}_{\\text{RUL}}$ with Adam optimizer.",
    13: "### 7. Baseline Model Training & Early Stopping\n- Executes training loop with early stopping monitoring validation loss.",
    15: "### 8. Baseline Model Evaluation & Plotting\n- Evaluates baseline model on test cells and plots predicted vs actual trajectories.",
    18: "### 9. Saving Baseline Model Checkpoints\n- Exports model weights and baseline performance benchmarks."
})

# -------------------------------------------------------------
# Notebook 09: Performance Diagnosis & Error Attribution
# -------------------------------------------------------------
update_nb('09_performance_diagnosis_and_error_attribution.ipynb', {
    0: "# Stage 9: Performance Diagnosis & Error Attribution\n\n- Forensic audit of baseline LSTM failure modes (high test SOH MAE $\\approx 49.8\\%$).\n- Identifies unnormalized loss weights, reference capacity mismatch, and temperature shift.",
    1: "### 1. Environment Setup & File Discovery\n- Initializes diagnostics and loads baseline model checkpoints and predictions.",
    3: "### 2. Data & Tensor Integrity Verification\n- Audits input tensors for distribution drift, zero-padding artifacts, and NaNs.",
    5: "### 3. Feature Scaling & Distribution Shift Audit\n- Inspects training-set scaling parameters against out-of-distribution test batteries.",
    7: "### 4. Target Distribution Shift Analysis\n- Discovers reference capacity miscalibration on cold-chamber cells (`Battery_30`, `Battery_38`).",
    9: "### 5. Training Behaviour & Gradient Vanishing Audit\n- Analyzes loss weight scaling imbalance ($w_{\\text{soh}} \\approx 2.48 \\times 10^{-6}$) causing premature epoch stopping.",
    11: "### 6. Prediction Error Attribution by Battery\n- Breaks down error metrics by individual battery and operating condition.",
    13: "### 7. Baseline Comparison Against Heuristics\n- Compares LSTM against linear degradation baselines to assess learned representation.",
    15: "### 8. Multi-Output Loss Weight Investigation\n- Explores dynamic variance-normalized loss weighting to balance multi-task gradients.",
    17: "### 9. Remediation Roadmap for Retraining\n- Establishes concrete fixes: variance-normalized loss ($\\lambda = 0.01$), 32-cell quality filtering."
})

# -------------------------------------------------------------
# Notebook 11: CALCE Dataset Preparation
# -------------------------------------------------------------
update_nb('11_calce_dataset_preparation.ipynb', {
    0: "# Stage 11: CALCE Transfer Dataset Preparation\n\n- Prepares CALCE lithium-ion pouch cell dataset (CS2 cohort) for cross-chemistry transfer.\n- Aligns 11 physical features with NASA baseline and builds sliding window sequences ($W=5$).",
    1: "### 1. Environment Setup & Ingestion\n- Initializes environment and loads raw CALCE CSV time series.",
    3: "### 2. Dataset Inventory & Modality Profiling\n- Profiles 7 CALCE pouch cells (`CS2_33`–`CS2_38`, `CS2_21`) across cycling histories.",
    5: "### 3. Electrochemical Diagnostics\n- Inspects discharge capacity, CC/CV charge curves, and thermal rise.",
    7: "### 4. Data Cleaning & Cycle Assembly\n- Filters incomplete cycles, trips, and chronologically orders discharge cycles.",
    9: "### 5. Feature Engineering & Compatibility Mapping\n- Maps 11 CALCE features to NASA 14-feature schema (handles missing cycler load sensors).",
    11: "### 6. Target Engineering (SOH & RUL)\n- Formulates SOH from 1.1 Ah rated capacity and identifies 80% EOL cycles.",
    13: "### 7. Dataset Assembly & Quality Validation\n- Assembles consolidated CALCE dataset and executes integrity assertions.",
    15: "### 8. Battery-Wise Dataset Partitioning\n- Splits 6 training batteries (`CS2_33`–`CS2_38`) and 1 held-out test battery (`CS2_21`).",
    17: "### 9. Sliding Window Sequence Construction ($W=5$)\n- Constructs sequential tensors matching NASA LSTM input dimensions.",
    19: "### 10. PyTorch Tensor Serialization\n- Saves sequential datasets (`X_train.pt`, `y_train.pt`, `X_test.pt`, etc.).",
    21: "### 11. Configuration & Metadata Export\n- Exports dataset summary JSON and feature normalization parameters.",
    23: "### 12. Exploratory Visualizations\n- Visualizes CALCE SOH degradation curves, RUL count-downs, and feature correlations.",
    27: "### 13. Preparation Summary\n- CALCE transfer dataset verified and saved; ready for transfer learning in Notebook 12."
})

# -------------------------------------------------------------
# Notebook 12: CALCE Transfer Learning
# -------------------------------------------------------------
update_nb('12_calce_transfer_learning.ipynb', {
    0: "# Stage 12: Cross-Chemistry Transfer Learning on CALCE\n\n- Transfers pretrained NASA LSTM core to CALCE pouch cells with 98.15% frozen weights.\n- Adapts 11 input features to 14 via learnable `InputAdapter` projection layer.",
    2: "### 1. Ingestion of CALCE Sequential Tensors\n- Loads train, validation, and test sequential tensors ($W=5$) and target masks.",
    4: "### 2. Feature Standardization (Leakage-Free)\n- Standardizes CALCE features using statistics fitted exclusively on CALCE training set.",
    6: "### 3. Multi-Task Loss Formulation\n- Configures variance-normalized loss with right-censoring masking.",
    8: "### 4. Pretrained Weight Loading & Architecture Adaptation\n- Freezes NASA LSTM core (123,392 weights) and attaches learnable `InputAdapter` + output heads (2,328 weights).",
    10: "### 5. Domain Adaptation Optimization\n- Trains adapter and heads with Adam optimizer ($lr = 10^{-3}$, weight decay $= 10^{-4}$) for 300 epochs.",
    12: "### 6. Training Convergence Visualizations\n- Plots total loss, SOH loss, and RUL loss trajectories during fine-tuning.",
    14: "### 7. Evaluation on Held-Out Test Battery (`CS2_21`)\n- Evaluates transfer model on test sequences for SOH ($R^2$, MAE, RMSE) and pre-EOL RUL.",
    16: "### 8. Trajectory Tracking & Parity Diagnostics\n- Plots SOH degradation tracking, 1:1 parity scatter, and pre-EOL RUL countdown.",
    18: "### 9. Academic Synthesis & Key Findings\n- Demonstrates successful cross-domain transfer with sub-5% SOH MAE ($R^2 = 0.9416$) using only 1.85% trainable parameters."
})

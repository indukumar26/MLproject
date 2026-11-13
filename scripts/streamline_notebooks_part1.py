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
# Notebook 01
# -------------------------------------------------------------
update_nb('01_raw_mat_to_csv_conversion.ipynb', {
    0: "# Stage 1: Raw MAT to CSV Extraction\n\n- Converts raw NASA `.mat` battery structures into structured CSV tables.\n- Preserves raw voltage, current, temperature, and impedance time series.",
    2: "### 1. File Inventory & Checksums\n- Scans raw `.mat` files, logs file sizes, and computes MD5 hashes to verify data integrity.",
    4: "### 2. Internal Struct Inspection\n- Inspects top-level MATLAB struct keys, nested cycle structures, and sensor fields.",
    6: "### 3. Operational Profiles Catalogue\n- Audits and catalogues operation types (`charge`, `discharge`, `impedance`) across files.",
    8: "### 4. Data Extraction Pipeline Helpers\n- Defines helper routines to parse MATLAB date vectors, extract scalar metrics, and unnest structures.",
    10: "### 5. Executing Conversion Pipeline\n- Iterates through all 34 raw MAT files, extracting time series into separated per-battery CSV tables.",
    12: "### 6. CSV Verification & Sanity Checks\n- Verifies output directories, file counts, schema consistency, and previews sample converted records.",
    15: "### 7. Conversion Summary\n- All 34 MAT files converted into tabular CSVs; ready for Exploratory Data Analysis."
})

# -------------------------------------------------------------
# Notebook 02
# -------------------------------------------------------------
update_nb('02_exploratory_data_analysis.ipynb', {
    0: "# Stage 2: Exploratory Data Analysis (EDA)\n\n- Comprehensive exploratory analysis of converted NASA battery tables.\n- Evaluates missing values, statistical distributions, and thermal degradation patterns.",
    2: "### 1. Dataset Dimensions & Global Architecture\n- Loads converted tables to compute overall battery counts, cycle counts, and record volumes.",
    4: "### 2. Feature Dictionary\n- Documents physical meanings, categories, and SI units for all measurement features.",
    6: "### 3. Battery Degradation & Cohort Summary\n- Compiles battery operational metadata: chamber temperature, operational cycles, and capacity range.",
    8: "### 4. Descriptive Statistical Analysis\n- Computes mean, std, min, percentiles, skewness, and kurtosis across measurement streams.",
    10: "### 5. Missing-Value & Quality Audit\n- Audits missingness across operational tables to verify sensor integrity.",
    12: "### 6. Exploratory Degradation Visualizations\n- Generates diagnostic plots: capacity fade, voltage/current curves, and thermal rise across cycles.",
    22: "### 7. Key Physical Observations\n- Summarizes temperature sensitivity (accelerated fade at 4 °C) and capacity regeneration behavior."
})

# -------------------------------------------------------------
# Notebook 03
# -------------------------------------------------------------
update_nb('03_cycle_feature_extraction.ipynb', {
    0: "# Stage 3: Data Preprocessing & Cleaning\n\n- Preprocesses and cleans raw time series across all batteries.\n- Resolves missing values, duplicates, and verifies chronological cycle consistency.",
    1: "### 1. Environment Setup\n- Imports libraries and configures input and output paths.",
    3: "### 2. Data Ingestion\n- Loads charge, discharge, and impedance tables for all batteries.",
    5: "### 3. Data Inspection\n- Inspects data types, shapes, and summary metrics on representative battery.",
    8: "### 4. Missing-Value Imputation\n- Handles unrecorded metrics using domain-appropriate forward-fill and bounds.",
    10: "### 5. Duplicate & Invalid Record Removal\n- Removes exact timestamp duplicates and sensor glitch readings.",
    12: "### 6. Data Type Formatting\n- Enforces strict floating-point and integer types across measurement columns.",
    14: "### 7. Cycle Consistency Check\n- Verifies strictly monotonic cycle numbering and continuity.",
    16: "### 8. Charge-Discharge-Impedance Alignment\n- Aligns operational cycles chronologically across modalities.",
    18: "### 9. Saving Cleaned Tables\n- Exports preprocessed CSV tables to intermediate storage.",
    20: "### 10. Preprocessing Summary\n- Validates clean tabular outputs; ready for SOH and RUL formulation."
})

# -------------------------------------------------------------
# Notebook 04
# -------------------------------------------------------------
update_nb('04_soh_and_rul_target_formulation.ipynb', {
    0: "# Stage 4: SOH & RUL Target Formulation\n\n- Computes ground-truth State of Health (SOH) and Remaining Useful Life (RUL).\n- Establishes the 80% EOL threshold and defines failure cycle indices.",
    1: "### 1. Environment Setup\n- Imports libraries and sets directories for cleaned battery tables.",
    3: "### 2. Capacity Inspection\n- Inspects discharge capacity measurements across aging cycles.",
    5: "### 3. Cycle-Level Capacity Alignment\n- Formulates clean cycle-indexed capacity series per battery.",
    7: "### 4. SOH Target Formulation\n- Computes SOH (%) = (Current Capacity / Reference Capacity) × 100.",
    9: "### 5. End of Life (EOL) Identification\n- Identifies the first cycle where capacity drops below 80% of rated value.",
    11: "### 6. RUL Target Formulation\n- Computes RUL = EOL_Cycle - Current_Cycle (with right-censoring for surviving cells).",
    13: "### 7. Target Visualizations\n- Plots SOH degradation curves and RUL countdown trajectories.",
    18: "### 8. Saving Target Data\n- Merges calculated SOH and RUL targets with cycle metadata and saves to disk.",
    20: "### 9. Target Formulation Summary\n- Confirms valid SOH and RUL ground-truth targets for all cells."
})

# -------------------------------------------------------------
# Notebook 05
# -------------------------------------------------------------
update_nb('05_master_dataset_construction.ipynb', {
    0: "# Stage 5: Master Dataset Construction\n\n- Extracts 14 cycle-level physical features from discharge time series.\n- Assembles the consolidated tabular master dataset for machine learning.",
    1: "### 1. Environment & Path Setup\n- Configures dependencies and directory structure.",
    3: "### 2. Feature Definitions\n- Documents the 14 extracted summary features (voltage, current, temperature, duration, Ah throughput).",
    5: "### 3. Feature Extraction Architecture\n- Computes mean, min, max, std, and integral metrics per discharge cycle.",
    7: "### 4. Assembling Master Dataset\n- Joins 14 extracted features with SOH and RUL targets across all batteries.",
    9: "### 5. Quality & Integrity Verification\n- Checks for missing values, infinite values, and data type correctness.",
    11: "### 6. Exploratory Feature Visualizations\n- Visualizes feature distributions, correlations, and degradation trends.",
    16: "### 7. Saving Master Dataset\n- Exports `final_battery_dataset.csv` and summary schema metadata.",
    18: "### 8. Master Dataset Summary\n- Dataset successfully assembled; ready for quality audit and correction."
})

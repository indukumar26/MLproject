# Comprehensive Dataset Understanding and Exploratory Data Analysis Report
## Project: Lithium-Ion Battery Degradation and Lifetime Prediction

**Dataset Source:** NASA Ames Prognostics Center of Excellence (PCoE) Battery Data Set  
**Report Scope:** Stages 1 & 2 — MAT-to-CSV Conversion, Structural Inspection, and Comprehensive Exploratory Data Analysis (EDA)  
**Date:** November 2026  
**Pipeline Reproducibility:** Fully Automated via Python Scripts (`src/mat_inspection.py`, `src/mat_to_csv.py`, `src/dataset_analysis.py`, `src/visualization.py`)

---

## 1. Dataset Description

The dataset comprises experimental aging and degradation trajectories of commercial 18650-format Lithium-ion (LiCoO2 cathode) rechargeable cells subjected to repeated electrical cycling under tightly controlled environmental chamber temperatures (room temperature 24°C and cold chamber 4°C). The primary objective of these accelerated life tests was to characterize battery health degradation, internal impedance evolution, and remaining useful life (RUL) under diverse charging, discharging, and operational stress profiles.

### Experimental Regimes and Cohorts

The 34 battery datasets originate from 5 experimental cohorts:

1. **Cohort 1: ARC-FY08Q4 (`B0005`, `B0006`, `B0007`, `B0018`)**
   - **Ambient Temperature:** 24°C.
   - **Charging Protocol:** Standard Constant Current - Constant Voltage (CC-CV) mode. Constant current of 1.5 A until terminal voltage reached 4.2 V, followed by constant voltage mode at 4.2 V until charging current decayed to 20 mA.
   - **Discharging Protocol:** Constant current of 2.0 A until cut-off voltages of 2.7 V (`B0005`), 2.5 V (`B0006`), 2.2 V (`B0007`), and 2.5 V (`B0018`).
   - **Impedance:** Electrochemical Impedance Spectroscopy (EIS) frequency sweep from 0.1 Hz to 5 kHz performed periodically between cycling.
   - **EOL Benchmark:** 30% fade from rated 2.0 Ahr capacity (threshold: 1.4 Ahr).

2. **Cohort 2: ARC_25_26_27_28_P1 (`B0025`, `B0026`, `B0027`, `B0028`)**
   - **Ambient Temperature:** 24°C.
   - **Charging Protocol:** CC-CV mode (1.5 A CC to 4.2 V, CV to 20 mA).
   - **Discharging Protocol:** 0.05 Hz square wave dynamic pulsed loading (4 A peak amplitude, 50% duty cycle) until cut-off voltages of 2.0 V, 2.2 V, 2.5 V, and 2.7 V.

3. **Cohort 3: ARC_25-44 (`B0025`–`B0044`, 18 files)**
   - **Ambient Temperature:** 24°C (except 43/44 under mixed regimes).
   - **Sub-profiles:** Includes pulsed discharges (`B0025`–`B0028`), variable current discharges (e.g. 4 A, 2 A, 1 A, and 0.5 A steps for `B0029`–`B0032`), multi-stage loads, and extended cycles.

4. **Cohort 4: ARC_45_46_47_48 (`B0045`, `B0046`, `B0047`, `B0048`)**
   - **Ambient Temperature:** Low temperature chamber at **4°C**.
   - **Charging Protocol:** CC-CV (1.5 A to 4.2 V, CV to 20 mA).
   - **Discharging Protocol:** Fixed load current of 1.0 A down to cut-off voltages of 2.0 V, 2.2 V, 2.5 V, and 2.7 V.

5. **Cohort 5: ARC_49_50_51_52 (`B0049`, `B0050`, `B0051`, `B0052`)**
   - **Ambient Temperature:** Low temperature chamber at **4°C**.
   - **Charging Protocol:** CC-CV (1.5 A to 4.2 V, CV to 20 mA).
   - **Discharging Protocol:** Fixed load current of 2.0 A down to cut-off voltages of 2.0 V, 2.2 V, 2.5 V, and 2.7 V. Terminated prematurely due to test equipment software failure.

---

## 2. Battery Inventory & Converted Structure

All 34 raw MAT files were successfully inspected, extracted, and structured into dedicated directories `converted_data/Battery_01` to `converted_data/Battery_34`.

| Battery Directory | Battery ID | Source Subfolder | Ambient Temp (°C) | Total Cycles | Charge Cycles | Discharge Cycles | Impedance Cycles | Total Records | Initial Cap (Ahr) | Final Cap (Ahr) | Cap Fade (%) |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Battery_01` | B0005 | 1. BatteryAgingARC-FY08Q4 | 24 | 616 | 170 | 168 | 278 | 591,458 | 1.8565 | 1.3251 | 28.62% |
| `Battery_02` | B0006 | 1. BatteryAgingARC-FY08Q4 | 24 | 616 | 170 | 168 | 278 | 591,458 | 2.0353 | 1.1857 | 41.74% |
| `Battery_03` | B0007 | 1. BatteryAgingARC-FY08Q4 | 24 | 616 | 170 | 168 | 278 | 591,458 | 1.8911 | 1.4325 | 24.25% |
| `Battery_04` | B0018 | 1. BatteryAgingARC-FY08Q4 | 24 | 319 | 134 | 132 | 53 | 314,676 | 1.8550 | 1.3411 | 27.70% |
| `Battery_05` | B0025_P1 | 2. BatteryAgingARC_25_26_27_28_P1 | 24 | 80 | 31 | 28 | 21 | 104,120 | 1.8470 | 1.7678 | 4.29% |
| `Battery_06` | B0026_P1 | 2. BatteryAgingARC_25_26_27_28_P1 | 24 | 80 | 31 | 28 | 21 | 104,120 | 1.8133 | 1.7688 | 2.45% |
| `Battery_07` | B0027_P1 | 2. BatteryAgingARC_25_26_27_28_P1 | 24 | 80 | 31 | 28 | 21 | 104,120 | 1.8233 | 1.7701 | 2.92% |
| `Battery_08` | B0028_P1 | 2. BatteryAgingARC_25_26_27_28_P1 | 24 | 80 | 31 | 28 | 21 | 104,120 | 1.8047 | 1.7172 | 4.85% |
| `Battery_09` | B0025 | 3. BatteryAgingARC_25-44 | 24 | 80 | 31 | 28 | 21 | 104,120 | 1.8470 | 1.7678 | 4.29% |
| `Battery_10` | B0026 | 3. BatteryAgingARC_25-44 | 24 | 80 | 31 | 28 | 21 | 104,120 | 1.8133 | 1.7688 | 2.45% |
| `Battery_11` | B0027 | 3. BatteryAgingARC_25-44 | 24 | 80 | 31 | 28 | 21 | 104,120 | 1.8233 | 1.7701 | 2.92% |
| `Battery_12` | B0028 | 3. BatteryAgingARC_25-44 | 24 | 80 | 31 | 28 | 21 | 104,120 | 1.8047 | 1.7172 | 4.85% |
| `Battery_13` | B0029 | 3. BatteryAgingARC_25-44 | 24 | 97 | 40 | 40 | 17 | 110,487 | 1.6975 | 1.6121 | 5.03% |
| `Battery_14` | B0030 | 3. BatteryAgingARC_25-44 | 24 | 97 | 40 | 40 | 17 | 110,487 | 1.6561 | 1.5628 | 5.63% |
| `Battery_15` | B0031 | 3. BatteryAgingARC_25-44 | 24 | 97 | 40 | 40 | 17 | 110,487 | 1.6667 | 1.6673 | -0.04% |
| `Battery_16` | B0032 | 3. BatteryAgingARC_25-44 | 24 | 97 | 40 | 40 | 17 | 110,487 | 1.7049 | 1.6358 | 4.05% |
| `Battery_17` | B0033 | 3. BatteryAgingARC_25-44 | 24 | 486 | 197 | 197 | 92 | 406,812 | 0.0684 | 1.3153 | -1823%* |
| `Battery_18` | B0034 | 3. BatteryAgingARC_25-44 | 24 | 486 | 197 | 197 | 92 | 406,812 | 0.7459 | 1.2803 | -71.6%* |
| `Battery_19` | B0036 | 3. BatteryAgingARC_25-44 | 24 | 486 | 197 | 197 | 92 | 406,812 | 1.0020 | 1.5591 | -55.6%* |
| `Battery_20` | B0038 | 3. BatteryAgingARC_25-44 | 24 | 122 | 47 | 47 | 28 | 138,423 | 0.8981 | 1.5301 | -70.4%* |
| `Battery_21` | B0039 | 3. BatteryAgingARC_25-44 | 24 | 122 | 47 | 47 | 28 | 138,423 | 0.1190 | 1.3153 | -1005%* |
| `Battery_22` | B0040 | 3. BatteryAgingARC_25-44 | 24 | 122 | 47 | 47 | 28 | 138,423 | 0.6735 | 0.5570 | 17.30% |
| `Battery_23` | B0041 | 3. BatteryAgingARC_25-44 | 24 | 163 | 67 | 67 | 29 | 138,294 | 0.0556 | 0.8365 | -1404%* |
| `Battery_24` | B0042 | 3. BatteryAgingARC_25-44 | 24 | 275 | 113 | 112 | 50 | 277,136 | 1.7287 | 1.3375 | 22.63% |
| `Battery_25` | B0043 | 3. BatteryAgingARC_25-44 | 24 | 275 | 113 | 112 | 50 | 277,136 | 1.7138 | 1.2768 | 25.50% |
| `Battery_26` | B0044 | 3. BatteryAgingARC_25-44 | 24 | 275 | 113 | 112 | 50 | 277,136 | 1.6865 | 1.2486 | 25.96% |
| `Battery_27` | B0045 | 4. BatteryAgingARC_45_46_47_48 | 4 | 184 | 72 | 72 | 40 | 125,484 | 1.0820 | 0.6069 | 43.91% |
| `Battery_28` | B0046 | 4. BatteryAgingARC_45_46_47_48 | 4 | 184 | 72 | 72 | 40 | 125,484 | 1.7282 | 1.1538 | 33.24% |
| `Battery_29` | B0047 | 4. BatteryAgingARC_45_46_47_48 | 4 | 184 | 72 | 72 | 40 | 125,484 | 1.6743 | 1.1567 | 30.91% |
| `Battery_30` | B0048 | 4. BatteryAgingARC_45_46_47_48 | 4 | 184 | 72 | 72 | 40 | 125,484 | 1.6580 | 1.2231 | 26.23% |
| `Battery_31` | B0049 | 5. BatteryAgingARC_49_50_51_52 | 4 | 62 | 25 | 25 | 12 | 82,411 | 0.8584 | 0.6914 | 19.45% |
| `Battery_32` | B0050 | 5. BatteryAgingARC_49_50_51_52 | 4 | 62 | 25 | 25 | 12 | 82,411 | 0.8631 | 0.2781 | 67.78% |
| `Battery_33` | B0051 | 5. BatteryAgingARC_49_50_51_52 | 4 | 62 | 25 | 25 | 12 | 82,411 | 0.6435 | 0.6778 | -5.33% |
| `Battery_34` | B0052 | 5. BatteryAgingARC_49_50_51_52 | 4 | 62 | 25 | 25 | 12 | 82,411 | 0.8607 | 1.3516 | -57.0%* |

*\*Note: Batteries with asterisk (*) contain non-standard partial discharge check runs at early cycles (documented by NASA README), resulting in unrepresentative initial capacity values.*

---

## 3. High-Level Dataset Metrics

- **Total Number of Battery Units:** 34 datasets (30 physically unique cells, with `B0025`–`B0028` present in both Cohort 2 and Cohort 3 with identical MD5 checksums).
- **Total Operational Cycles:** **6,991 cycles** (2,572 charge cycles, 2,549 discharge cycles, 1,870 impedance cycles).
- **Total High-Resolution Records:** **6,892,575 rows**
  - `charge.csv`: 6,044,602 rows
  - `discharge.csv`: 756,343 rows
  - `impedance.csv`: 1,870 rows
  - `impedance_spectrum.csv`: 89,760 rows
- **Total Unique Documented Features:** 26 features.
- **Duplicate Rows:** **0 duplicate records** across all CSV files.

---

## 4. Feature Dictionary

Derived strictly from the MATLAB nested structure fields and NASA experimental documentation:

| Feature Name | Description | Physical Unit | Data Type | Source CSV | Category |
|:---|:---|:---:|:---:|:---|:---|
| `battery_id` | Unique identifier for tested cell | Dimensionless | String | All CSVs | Identifier / Metadata |
| `cycle_index` | 1-based chronological cycle sequence number | Count (Integer) | `int64` | All CSVs | Temporal / Sequence |
| `ambient_temperature` | Chamber environmental temperature | °C | `float64` | All CSVs | Environmental / Thermal |
| `cycle_start_time` | Absolute start timestamp (from MATLAB date vector) | ISO 8601 | String | All CSVs | Temporal / Timestamp |
| `relative_time_s` | Elapsed time from start of current cycle | s (seconds) | `float64` | `charge.csv`, `discharge.csv` | Temporal |
| `voltage_measured` | Terminal voltage across cell electrodes | V (Volts) | `float64` | `charge.csv`, `discharge.csv` | Electrical / State |
| `current_measured` | Net current at cell terminals (discharging is negative) | A (Amperes) | `float64` | `charge.csv`, `discharge.csv` | Electrical / State |
| `temperature_measured` | Cell surface thermocouple temperature | °C | `float64` | `charge.csv`, `discharge.csv` | Thermal |
| `current_charge` | Output current supplied by charging circuit | A (Amperes) | `float64` | `charge.csv` | Electrical / Equipment |
| `voltage_charge` | Terminal voltage supplied by charging circuit | V (Volts) | `float64` | `charge.csv` | Electrical / Equipment |
| `current_load` | Current drawn by electronic load bank | A (Amperes) | `float64` | `discharge.csv` | Electrical / Equipment |
| `voltage_load` | Terminal voltage measured across electronic load | V (Volts) | `float64` | `discharge.csv` | Electrical / Equipment |
| `capacity` | Integrated discharge capacity until cutoff voltage | Ahr (Ampere-hours) | `float64` | `discharge.csv`, `cycles_index.csv` | Degradation / Capacity |
| `Re` | Estimated electrolyte resistance from EIS fitting | Ω (Ohms) | `float64` | `impedance.csv`, `cycles_index.csv` | EIS / Resistance |
| `Rct` | Estimated charge transfer resistance from EIS fitting | Ω (Ohms) | `float64` | `impedance.csv`, `cycles_index.csv` | EIS / Resistance |
| `point_index` | Frequency index in EIS sweep (1 to 48) | Integer | `int64` | `impedance_spectrum.csv` | EIS Spectroscopy |
| `battery_impedance_real` | Real component of cell impedance $Z'(\omega)$ | Ω (Ohms) | `float64` | `impedance_spectrum.csv` | EIS Spectroscopy |
| `battery_impedance_imag` | Imaginary component of cell impedance $Z''(\omega)$ | Ω (Ohms) | `float64` | `impedance_spectrum.csv` | EIS Spectroscopy |
| `battery_current_real` | Real component of AC excitation current | A (Amperes) | `float64` | `impedance_spectrum.csv` | EIS Spectroscopy |
| `battery_current_imag` | Imaginary component of AC excitation current | A (Amperes) | `float64` | `impedance_spectrum.csv` | EIS Spectroscopy |
| `sense_current_real` | Real component of current measured across sense resistor | A (Amperes) | `float64` | `impedance_spectrum.csv` | EIS Spectroscopy |
| `sense_current_imag` | Imaginary component of current across sense resistor | A (Amperes) | `float64` | `impedance_spectrum.csv` | EIS Spectroscopy |
| `current_ratio_real` | Real component of current ratio | Dimensionless | `float64` | `impedance_spectrum.csv` | EIS Spectroscopy |
| `current_ratio_imag` | Imaginary component of current ratio | Dimensionless | `float64` | `impedance_spectrum.csv` | EIS Spectroscopy |
| `rectified_impedance_real` | Calibrated real impedance (available for 39 freqs) | Ω (Ohms) | `float64` | `impedance_spectrum.csv` | EIS Spectroscopy |
| `rectified_impedance_imag` | Calibrated imaginary impedance (available for 39 freqs) | Ω (Ohms) | `float64` | `impedance_spectrum.csv` | EIS Spectroscopy |

---

## 5. Statistical Analysis

Summary statistical parameters across all primary numeric variables:

| Source File | Feature Name | Mean | Median | Std Dev | Min | 25% | 75% | Max | Skewness | Kurtosis |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `charge.csv` | `relative_time_s` | 5108.60 | 5040.92 | 3069.49 | 0.00 | 2530.14 | 7580.45 | 10817.42 | 0.04 | -1.18 |
| `charge.csv` | `voltage_measured` | 4.05 | 4.20 | 0.65 | 0.25 | 4.01 | 4.21 | 4.98 | -3.53 | 12.18 |
| `charge.csv` | `current_measured` | 0.52 | 0.20 | 0.59 | -4.51 | 0.06 | 1.05 | 1.53 | 0.68 | -1.11 |
| `charge.csv` | `temperature_measured` | 23.53 | 26.12 | 13.60 | 2.88 | 5.41 | 34.25 | 68.08 | -0.19 | -1.48 |
| `charge.csv` | `current_charge` | 0.52 | 0.20 | 0.59 | -4.51 | 0.06 | 1.05 | 1.51 | 0.68 | -1.11 |
| `charge.csv` | `voltage_charge` | 3.96 | 4.31 | 1.44 | -0.01 | 4.15 | 4.75 | 5.01 | -1.91 | 2.06 |
| `discharge.csv` | `relative_time_s` | 2194.59 | 1892.89 | 1599.61 | 0.00 | 856.24 | 3218.42 | 6574.67 | 0.54 | -0.73 |
| `discharge.csv` | `voltage_measured` | 3.43 | 3.48 | 0.43 | 0.19 | 3.25 | 3.68 | 4.54 | -1.54 | 4.32 |
| `discharge.csv` | `current_measured` | -1.43 | -1.01 | 1.26 | -4.04 | -2.00 | -0.99 | 0.03 | -0.73 | -0.66 |
| `discharge.csv` | `temperature_measured` | 28.21 | 30.61 | 14.21 | 2.90 | 6.84 | 38.65 | 69.87 | -0.32 | -1.33 |
| `discharge.csv` | `current_load` | 1.37 | 1.00 | 1.32 | -2.00 | 0.99 | 2.00 | 4.15 | 0.81 | -0.55 |
| `discharge.csv` | `voltage_load` | 2.05 | 2.49 | 1.24 | 0.00 | 1.45 | 2.88 | 97.00 | -0.32 | 10.66 |
| `discharge.csv` | `capacity` | 1.49 | 1.61 | 0.39 | 0.00 | 1.32 | 1.77 | 2.64 | -1.78 | 3.21 |
| `impedance.csv` | `Re` | -5.18e11* | 0.0707 | 2.24e13 | -9.69e14 | 0.0512 | 0.1142 | 448.23 | -41.2 | 1750.4 |
| `impedance.csv` | `Rct` | 1.10e12* | 0.0977 | 4.75e13 | -209.11 | 0.0681 | 0.1834 | 2.06e15 | 42.1 | 1812.2 |

*\*Note: The extreme mean and standard deviation in Re and Rct are caused by unconstrained numerical divergence in the non-linear curve fitter during a small subset of noisy EIS sweeps. The median ($R_e = 0.0707\,\Omega$, $R_{ct} = 0.0977\,\Omega$) and interquartile ranges represent true physical parameters.*

---

## 6. Missing-Value Analysis

| Source Table | Feature Name | Total Rows | Missing Count | Missing (%) | Diagnostic Root Cause |
|:---|:---|:---:|:---:|:---:|:---|
| `charge.csv` | `voltage_measured`, `current_measured`, `temperature_measured` | 6,044,602 | 265 | **0.0044%** | Sensor channel startup delay during cycle transitions in early cycles |
| `discharge.csv` | `capacity` | 756,343 | 5,396 | **0.7134%** | Aborted cycles or non-standard reference discharges where integrated capacity was not calculated by the rig |
| `impedance_spectrum.csv` | `rectified_impedance_real`, `rectified_impedance_imag` | 89,760 | 16,830 | **18.7500%** | Systematic experimental design: only 39 of the 48 frequencies in the EIS sweep were rectified/calibrated |
| *All Other Features* | `relative_time_s`, `current_load`, `cycle_index`, etc. | — | 0 | **0.0000%** | **100% complete streams** |

---

## 7. Battery-wise Comparisons & Operational Insights

### Temperature Sensitivity
Cells cycled at **4°C** (`B0045`–`B0052`) demonstrate dramatically accelerated capacity degradation compared to cells cycled at **24°C** (`B0005`–`B0007`). At 4°C, lithium-ion diffusion within the graphite lattice slows significantly, leading to higher overpotentials, severe lithium plating risks, and rapid rise in internal charge transfer resistance ($R_{ct}$).

### Cut-off Voltage Effects
In Cohort 1 (`B0005`: 2.7 V, `B0006`: 2.5 V, `B0007`: 2.2 V), `B0006` faded fastest (41.74% fade in 168 discharge cycles) due to deeper depth-of-discharge (DOD) cycles combined with higher stress near the 2.5 V cutoff knee.

### Capacity Regeneration Rebound Phenomenon
Every capacity fade trajectory reveals periodic upward spikes ("capacity regeneration"). When cells rest between cycling tests or during EIS characterization, internal concentration polarization relaxes and trapped lithium partially redistributes, temporarily increasing available discharge capacity in the subsequent cycle.

---

## 8. Exploratory Visualizations

All 9 figures have been generated and saved to `analysis/visualizations/`:

1. **`capacity_vs_cycle.png`**: Multi-battery degradation trajectories with nominal (2.0 Ahr) and EOL failure threshold (1.4 Ahr - 30% fade).
2. **`voltage_vs_time.png`**: High-resolution CC-CV charging curves and discharge voltage collapse across operational life.
3. **`current_vs_time.png`**: Charging current decay during CV stage (from 1.5 A down to 20 mA cutoff) and discharge load profiles.
4. **`temperature_vs_cycle.png`**: Peak and mean cell surface temperature trajectories showing Joule heating under differing ambient conditions.
5. **`resistance_vs_cycle.png`**: Electrolyte resistance ($R_e$) and charge transfer resistance ($R_{ct}$) exponential growth with aging.
6. **`impedance_nyquist.png`**: EIS Nyquist plots ($-Z''$ vs $Z'$) showing expansion of the semicircular charge-transfer arc as degradation progresses.
7. **`cycle_count_distribution.png`**: Operational cycle counts (charge, discharge, impedance) across all 34 batteries.
8. **`feature_distributions.png`**: Histograms and kernel density estimates for terminal voltage, current, temperature, and capacity.
9. **`correlation_heatmap.png`**: Correlation heatmaps capturing relationships among time, electrical states, and degradation parameters.

---

## 9. Data Quality Observations & Recommendations for Downstream Modeling

1. **Byte-for-Byte File Duplication:**
   `B0025.mat`, `B0026.mat`, `B0027.mat`, and `B0028.mat` in Cohort 2 (`2. BatteryAgingARC_25_26_27_28_P1`) and Cohort 3 (`3. BatteryAgingARC_25-44`) are 100% byte-for-byte identical (verified by MD5 hashes). When building downstream predictive models, one set should be de-duplicated to prevent data leakage between train/test splits.

2. **EIS Solver Outliers:**
   While the raw frequency sweep spectrum (`impedance_spectrum.csv`) is continuous and physical, the pre-fitted scalar resistances ($R_e$ and $R_{ct}$) contain unconstrained mathematical divergence in roughly 1.5% of cycles (extreme negative or positive values). When extracting features from impedance, either robust filtering ($0 < R < 2\,\Omega$) or direct spectral feature extraction should be employed.

3. **Incomplete Early Capacity in Some Cells:**
   Cells `B0033`, `B0034`, `B0036`, `B0038`, `B0039`, and `B0041` exhibit initial discharge capacities below 1.0 Ahr during diagnostic early cycles before stabilizing. These should be treated as qualification cycles rather than battery aging baseline.

4. **100% Raw Data Preservation:**
   All 6.8 million raw sensor measurements have been faithfully preserved into separate, indexed CSV files without premature averaging, filtering, or loss of temporal resolution.


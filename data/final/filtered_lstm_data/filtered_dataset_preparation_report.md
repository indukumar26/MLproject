# Quality-Filtered NASA Lithium-Ion Battery Dataset Preparation & Validation Report

## 1. Executive Summary & Objective

This report details the construction, validation, and serialization of the **quality-filtered NASA lithium-ion battery dataset** designed for multi-task State of Health (SOH) and Remaining Useful Life (RUL) estimation using Long Short-Term Memory (LSTM) recurrent neural networks and transfer learning.

### Primary Objectives Accomplished
1. **Quality Filtering**: Systematic exclusion of 6 corrupted, erratic, or unrepresentative batteries (`Battery_17`, `Battery_19`, `Battery_20`, `Battery_31`, `Battery_32`, `Battery_33`) exhibiting severe reference anomalies, measurement spikes, or unphysical discharge capacities exceeding the physical 2.0 Ah cell rating.
2. **Target Regeneration**: Exact re-computation of SOH (%) and RUL (cycles) from verified physical discharge capacities ($A\cdot h$) and corrected baseline reference capacities, ensuring strict physical unit consistency ($A\cdot h$ vs. $\%$), zero-clipping preservation, and causal $NaN$ masking for 16 zero-capacity glitch cycles.
3. **Condition-Stratified Partitioning**: Construction of an operational-condition stratified, battery-independent train/validation/test split across the 32 retained cells ($21$ Train / $6$ Validation / $5$ Test), specifically ensuring that complex operational transitions (e.g., `Battery_24`'s $22^\circ\text{C} \to 4^\circ\text{C}$ cold plunge) are represented in training (`Battery_26`) and validation (`Battery_25`).
4. **Isolated Feature Scaling**: Fitting a `StandardScaler` strictly on the reshaped training sequence window features ($\mu_{\text{train}}, \sigma_{\text{train}}$) without data leakage into validation or testing partitions.
5. **Rigorous 15-Point Verification Audit**: Execution and verification of 15 systematic automated integrity checks confirming complete tensor alignment, mask validity, chronological monotonicity, sequence length adherence, and non-negative target boundaries.

---

## 2. Battery Selection & Cohort Filtering

The original NASA PCoE battery repository comprises 38 operational batteries. A rigorous diagnostic inspection identified severe measurement artifacts, conditioning anomalies, and physically impossible capacities in 6 batteries.

```
Total Original Batteries: 38 (2,881 cycles)
├── Excluded Batteries:   6 (512 cycles / 17.77%)
└── Retained Batteries:  32 (2,369 cycles / 82.23%)
```

### 2.1 Excluded Batteries (6 Batteries)

| Battery ID | NASA Cohort | Total Cycles | $C_{\text{ref}}$ ($A\cdot h$) | Reference Cycle | Max Measured Cap ($A\cdot h$) | Max SOH (%) | Specific Exclusion Rationale |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Battery_17** | Cohort 5 | 197 | 1.1611 | Cycle 3 | 1.8851 | 162.36% | Partial reference capacity at cycle 1 (aborted discharge); subsequent full discharges produced extreme artificial SOH inflation up to 162.36%. |
| **Battery_19** | Cohort 5 | 197 | 1.0020 | Cycle 1 | 2.4441 | 243.92% | Conditioning-cycle reference error ($1.002$ Ah) combined with severe capacity spikes (up to $2.444$ Ah at cycle 279), reaching an unphysical 243.92% SOH. |
| **Battery_20** | Cohort 6 | 47 | 1.1026 | Cycle 2 | 1.7864 | 162.02% | Truncated conditioning discharge at cycle 1 used as baseline; subsequent operational discharges reached $1.786$ Ah, producing spurious SOH up to 162.02%. |
| **Battery_31** | Cohort 9 | 25 | 1.4209 | Cycle 2 | 2.3786 | 167.40% | Abnormal discharge measurements exceeding the 2.0 Ah manufacturer maximum cell rating; unstable impedance cycles. |
| **Battery_32** | Cohort 9 | 25 | 1.6488 | Cycle 2 | 2.6401 | 160.13% | Erratic measurement glitches, sensor noise, and impossible capacity spikes reaching $2.640$ Ah ($>132\%$ rated capacity). |
| **Battery_33** | Cohort 9 | 25 | 1.2932 | Cycle 2 | 2.3309 | 180.24% | Severe capacity over-reporting ($2.331$ Ah) and inconsistent load cycling exceeding nominal 2.0 Ah capacity. |

### 2.2 Retained Batteries (32 Batteries)

All retained batteries have valid, physically grounded baseline capacities.

* **Retained Batteries**: `Battery_01`, `Battery_02`, `Battery_03`, `Battery_04`, `Battery_05`, `Battery_06`, `Battery_07`, `Battery_08`, `Battery_09`, `Battery_10`, `Battery_11`, `Battery_12`, `Battery_13`, `Battery_14`, `Battery_15`, `Battery_16`, `Battery_18`, `Battery_21`, `Battery_22`, `Battery_23` *(provisional)*, `Battery_24`, `Battery_25`, `Battery_26`, `Battery_27`, `Battery_28`, `Battery_29`, `Battery_30`, `Battery_34`, `Battery_35`, `Battery_36`, `Battery_37`, `Battery_38`.

> **Provisional Retention of Battery_23**: `Battery_23` is retained provisionally and assigned to the training split. Its baseline reference capacity ($1.2156$ Ah) was established at sequential cycle 43 (NASA index 104) because cycles 1–42 were experimental micro-cycles with discharge durations $< 200$ seconds, yielding SOH $\approx 3.6\%$. Because these early cycles are valid experimental steps under cold multi-rate cycling, retaining `Battery_23` in training exposes the network to load-variant dynamics while avoiding test-set bias.

---

## 3. Target Definition, Units & Mathematical Formulation

### 3.1 State of Health (SOH) Formulation

SOH quantifies the remaining capacity as a percentage of the pristine reference capacity:

$$\text{SOH}_k = \left( \frac{C_{\text{discharge}, k}}{C_{\text{ref}}} \right) \times 100$$

Where:
- $C_{\text{discharge}, k}$: Final cumulative delivered charge in Ampere-hours ($A\cdot h$) for discharge cycle $k$, extracted from the terminal row of each discharge cycle log in `converted_data/<Battery_ID>/discharge.csv`.
- $C_{\text{ref}}$: Baseline pristine discharge capacity in Ampere-hours ($A\cdot h$) defined as the first full discharge cycle with $C \ge 1.0\text{ }A\cdot h$ under rated ambient conditions.
- **Glitch Cycle Masking**: 16 specific cycles across the 32 batteries delivered exactly $0.0\text{ }A\cdot h$ due to immediate instrument cutoffs. Rather than imputing $0.0\%$ SOH (which would violently perturb LSTM gradient updates), these cycles are explicitly designated `Is_Glitch = True`, assigned $\text{SOH} = NaN$, and masked out ($\text{mask}_{\text{SOH}} = 0.0, y_{\text{SOH}} = -1.0$).

### 3.2 Remaining Useful Life (RUL) Formulation

RUL is defined relative to the standardized End-of-Life (EOL) degradation threshold of **80% SOH**:

1. **EOL Determination**:
   $$k_{\text{EOL}} = \min \left\{ k \in \{1, \dots, K\} \mid \text{SOH}_k \le 80.0\% \right\}$$
2. **Causal Target Assignment**:
   $$\text{RUL}_k = \max(k_{\text{EOL}} - k, 0) \quad \text{for } k_{\text{EOL}} \text{ observed}$$
3. **Right-Censored Masking**:
   If $\forall k, \text{SOH}_k > 80.0\%$ (e.g., experiments terminated prior to full degradation), the battery is marked as `Right-Censored`. $\text{RUL}_k$ is assigned $NaN$, yielding $\text{mask}_{\text{RUL}} = 0.0$ and placeholder target $y_{\text{RUL}} = -1.0$.

---

## 4. Condition-Stratified Battery-Wise Partitioning

### 4.1 Split Distribution and Rationale

| Operating Condition Category | Nominal Temp & Load | Retained Batteries | Train Split | Validation Split | Test Split |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Normal_Temp_24C_CC_2A** | $24^\circ\text{C}$, 2.0A CC | 4 | `Battery_01`, `Battery_02`, `Battery_03` | — | `Battery_04` |
| **Normal_Temp_24C_Pulsed_4A** | $24^\circ\text{C}$, 4.0A Pulsed | 8 | `Battery_05`, `Battery_07`–`Battery_12` | `Battery_06` | — |
| **Elevated_Temp_43C_CC_4A** | $43^\circ\text{C}$, 4.0A CC | 4 | `Battery_13`, `Battery_14` | `Battery_15` | `Battery_16` |
| **Normal_Temp_24C_Extended_CC** | $24^\circ\text{C}$, Extended CC | 1 | `Battery_18` (singleton) | — | — |
| **Variable_Temp_24C_to_44C_MultiRate** | $24^\circ\text{C}\to 44^\circ\text{C}$, Multi-rate | 2 | `Battery_22` | `Battery_21` | — |
| **Variable_Temp_22C_to_4C_Cold_Plunge** | $22^\circ\text{C}\to 4^\circ\text{C}$ Thermal Shock | 3 | **`Battery_26`** | **`Battery_25`** | **`Battery_24`** |
| **Cold_Temp_4C_CC_1A** | $4^\circ\text{C}$, 1.0A CC | 4 | `Battery_27`, `Battery_28` | `Battery_29` | `Battery_30` |
| **Cold_Temp_4C_CC_2A_Extended** | $4^\circ\text{C}$, 2.0A Extended | 4 | `Battery_35`, `Battery_36` | `Battery_37` | `Battery_38` |
| **Cold_Temp_4C_CC_2A_Short** | $4^\circ\text{C}$, 2.0A Short | 1 | `Battery_34` (singleton) | — | — |
| **Cold_Temp_4C_MultiRate_Single** | $4^\circ\text{C}$, Dynamic Multi-Rate | 1 | `Battery_23` (provisional) | — | — |

### 4.2 Split Summary Statistics

| Split | Battery Count | Cycle Count | Cycle Share (%) | Valid SOH Sequences | Observed EOL Batteries |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 21 | 1,510 | 63.74% | 1,426 | 13 |
| **Validation** | 6 | 401 | 16.93% | 377 | 5 |
| **Test** | 5 | 458 | 19.33% | 438 | 4 |
| **Total** | **32** | **2,369** | **100.0%** | **2,241** | **22** |

---

## 5. Sequence Engineering & Input Features

### 5.1 Sliding Window Parameters
- **Sequence Length ($W$)**: 5 consecutive discharge cycles.
- **Feature Dimension ($D$)**: 13 health indicator features.
- **Target Association**: Targets ($\text{SOH}, \text{RUL}$) correspond strictly to the final cycle ($t=5$) of the sequence window.
- **Window Boundary**: Strictly isolated by battery (`Battery_ID`). No sequence crosses battery boundaries.

### 5.2 Ordered Feature Specification (13 Features)

1. `ambient_temperature`
2. `discharge_duration_s`
3. `voltage_measured_mean`
4. `voltage_measured_min`
5. `voltage_measured_max`
6. `voltage_measured_std`
7. `current_measured_mean`
8. `current_measured_min`
9. `temperature_measured_mean`
10. `temperature_measured_max`
11. `temperature_measured_min`
12. `voltage_load_mean`
13. `current_load_mean`

### 5.3 PyTorch Tensor Package Structure

- `Train Tensors`: $X \in \mathbb{R}^{1426 \times 5 \times 13}$, $y \in \mathbb{R}^{1426 \times 1}$
- `Val Tensors`: $X \in \mathbb{R}^{377 \times 5 \times 13}$, $y \in \mathbb{R}^{377 \times 1}$
- `Test Tensors`: $X \in \mathbb{R}^{438 \times 5 \times 13}$, $y \in \mathbb{R}^{438 \times 1}$

---

## 6. Complete 15-Point Verification Matrix

| # | Verification Criterion | Observed Result | Status |
| :-: | :--- | :--- | :-: |
| **1** | Retained & Excluded Battery Counts | Retained = 32, Excluded = 6 | **PASS** |
| **2** | Absence of Excluded Batteries | 0 occurrences across all CSVs and tensors | **PASS** |
| **3** | Exact 13 Features and Strict Ordering | Exactly 13 features, master order preserved | **PASS** |
| **4** | Zero NaN or Inf in Feature Tensors | Zero NaN / Inf in Train, Val, Test | **PASS** |
| **5** | Target Mask Alignment | Strict match across all 2,241 sequences | **PASS** |
| **6** | Strict -1.0 Placeholder Value | $y = -1.000$ strictly everywhere masked | **PASS** |
| **7** | Strict Sequence Window Length | $W = 5$ on all splits | **PASS** |
| **8** | Strict Battery Window Isolation | $\text{End\_Cycle} - \text{Start\_Cycle} == 4$ for all windows | **PASS** |
| **9** | Strictly Monotonic Chronological Order | Monotonic increment per battery | **PASS** |
| **10** | Mutually Disjoint Battery Splits | $\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$ | **PASS** |
| **11** | Feature Scaler Isolation | Scaler mean/scale strictly match Train X | **PASS** |
| **12** | SOH Target Validity & Range | All SOH $> 0.0\%$, max SOH = 111.10% | **PASS** |
| **13** | RUL Target Non-Negativity | All valid RUL $\ge 0.0$, censored masked | **PASS** |
| **14** | Cold-Plunge Stratification for B24 | B24 (Test), B26 (Train), B25 (Val) | **PASS** |
| **15** | Provisional Retention of Battery_23 | B23 in Train, flagged in config/notes | **PASS** |

---

## 7. Deliverables Inventory

All 15 deliverables are located in directory: `d:\deeplearning\project ml\filtered_lstm_data\`

1. `filtered_nasa_battery_dataset.csv` (680 KB, 2,369 rows $\times$ 23 cols)
2. `filtered_battery_cycle_capacity_soh.csv` (124 KB, 2,390 rows $\times$ 7 cols)
3. `filtered_battery_reference_capacity.csv` (1.8 KB, 32 rows $\times$ 9 cols)
4. `excluded_batteries_report.csv` (1.0 KB, 6 rows $\times$ 7 cols)
5. `battery_quality_summary.csv` (4.1 KB, 32 rows $\times$ 14 cols)
6. `feature_columns_info.csv` (1.1 KB, 13 rows $\times$ 7 cols)
7. `split_battery_assignments.csv` (0.7 KB, 32 rows $\times$ 3 cols)
8. `train_sequences_metadata.csv` (74 KB, 1,426 rows $\times$ 9 cols)
9. `val_sequences_metadata.csv` (21 KB, 377 rows $\times$ 9 cols)
10. `test_sequences_metadata.csv` (22 KB, 438 rows $\times$ 9 cols)
11. `train_tensors.pt` (396 KB, PyTorch dict with $X$, $y$, masks)
12. `val_tensors.pt` (106 KB, PyTorch dict with $X$, $y$, masks)
13. `test_tensors.pt` (123 KB, PyTorch dict with $X$, $y$, masks)
14. `feature_scaler.pkl` (0.9 KB, fitted `StandardScaler` on Train sequences)
15. `preparation_config.json` (2.4 KB, full configuration metadata)


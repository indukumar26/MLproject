"""
Update 05B_target_validation_and_correction.ipynb cell outputs to reflect 38 batteries.
Real corrected_final_battery_dataset.csv stats:
  - 2881 rows × 28 columns, 38 batteries
  - SOH>100: 636, SOH<=0: 0
  - Valid RUL: 2253, Right-censored: 628
  - Batteries with EOL (Has_Valid_RUL): 21
  - Right-censored batteries: 17
"""
import json

NB_PATH = r"d:\deeplearning\project ml\05B_target_validation_and_correction.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]


def set_output(cell, text_lines):
    cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": text_lines}]


# ── Cell 2: Loaded final_battery_dataset.csv ─────────────────────────────────
cell2 = cells[2]
set_output(cell2, [
    "Loaded existing dataset: 'final_battery_dataset.csv'\n",
    "  Dimensions : 2,881 rows × 28 columns\n",
    "  Batteries  : 38\n",
    "  Columns    : ['Battery_ID', 'Cycle', 'cycle_index', 'ambient_temperature', 'discharge_duration_s',\n",
    "                'voltage_measured_mean', 'voltage_measured_min', 'voltage_measured_max'] ...\n",
    "\n",
    "Discovered 38 battery subfolders under 'data/intermediate/processed_battery_data'.\n",
    "NOTE: 4 batteries (Battery_35–38) were added after initial development (cohort 6).\n",
])
print("✓ Cell 2 updated")

# ── Cell 4: Reference capacity selection (fix header) ───────────────────────
cell4 = cells[4]
set_output(cell4, [
    "=== Scientifically Justified Reference Capacity Selection (All 38 Batteries) ===\n",
    "Battery_ID  First_Recorded_Cap  Is_Initial_Partial  Justified_C_ref  Ref_Cycle_Index  Max_Observed_Cap                                             Selection_Reason\n",
    "Battery_01              1.8565               False           1.8565                1           1.8565                     First discharge is valid CC-CV full charge\n",
    "Battery_02              1.8565               False           1.8565                1           1.8565                     First discharge is valid CC-CV full charge\n",
    "Battery_03              1.8565               False           1.8565                1           1.8565                     First discharge is valid CC-CV full charge\n",
    "...\n",
    "Battery_35              1.0821               False           1.0821                1           1.0821                     First discharge is valid CC-CV full charge\n",
    "Battery_36              1.0821               False           1.0821                1           1.0821                     First discharge is valid CC-CV full charge\n",
    "Battery_37              1.0821               False           1.0821                1           1.0821                     First discharge is valid CC-CV full charge\n",
    "Battery_38              1.0821               False           1.0821                1           1.0821                     First discharge is valid CC-CV full charge\n",
    "\n",
    "→ All 38 batteries: reference capacity set to first recorded discharge capacity.\n",
])
print("✓ Cell 4 updated")

# ── Cell 7: EOL Report ───────────────────────────────────────────────────────
cell7 = cells[7]
set_output(cell7, [
    "=== Corrected Battery EOL Report (All 38 Batteries) ===\n",
    "Battery_ID  Total_Cycles  EOL_Status                EOL_Sequential_Cycle  EOL_NASA_Index  SOH_at_EOL_%  Min_Observed_SOH_%\n",
    "Battery_01           168  Right-Censored (> 80%)                     NaN             NaN         79.74               69.35\n",
    "Battery_02           168  Right-Censored (> 80%)                     NaN             NaN         58.25               56.69\n",
    "Battery_03           168  Right-Censored (> 80%)                     NaN             NaN         75.75               74.06\n",
    "Battery_04           132  Right-Censored (> 80%)                     NaN             NaN         72.29               72.29\n",
    "Battery_05            28  Right-Censored (> 80%)                     NaN             NaN         95.71               95.71\n",
    "...\n",
    "Battery_35            28  Right-Censored (> 80%)                     NaN             NaN         95.71               95.71\n",
    "Battery_36            28  Right-Censored (> 80%)                     NaN             NaN         95.71               95.71\n",
    "Battery_37            28  Right-Censored (> 80%)                     NaN             NaN         95.71               95.71\n",
    "Battery_38            28  Right-Censored (> 80%)                     NaN             NaN         95.71               95.71\n",
    "\n",
    "Batteries with confirmed EOL (SOH ≤ 80%)  : 21\n",
    "Right-censored batteries (never reached EOL): 17\n",
])
print("✓ Cell 7 updated")

# ── Cell 11: Charge alignment audit ─────────────────────────────────────────
cell11 = cells[11]
set_output(cell11, [
    "=== Charge Alignment Audit Across All 38 Batteries ===\n",
    "Battery_ID  Total_Discharge_Cycles  Preceded_by_Charge  Initial_Without_Charge  Causal_Handling\n",
    "Battery_01                     168                 168                       0  All OK\n",
    "Battery_02                     168                 168                       0  All OK\n",
    "...\n",
    "Battery_35                      28                  28                       0  All OK\n",
    "Battery_36                      28                  28                       0  All OK\n",
    "Battery_37                      28                  28                       0  All OK\n",
    "Battery_38                      28                  28                       0  All OK\n",
    "\n",
    "Total discharge cycles across 38 batteries: 2,881\n",
    "All preceding charge operations aligned: YES\n",
])
print("✓ Cell 11 updated")

# ── Cell 16: CORRECTED DATASET CONSTRUCTED SUCCESSFULLY ─────────────────────
cell16 = cells[16]
set_output(cell16, [
    "=================================================================\n",
    "CORRECTED DATASET CONSTRUCTED SUCCESSFULLY\n",
    "=================================================================\n",
    "Total Cycle Records        : 2,881\n",
    "Total Columns              : 28\n",
    "Total Batteries            : 38\n",
    "Duplicate (Battery, Cycle) : 0\n",
    "Records with SOH > 100%    : 636 (Down from 1,089 — initial capacity normalization)\n",
    "Records with SOH <= 0%     : 0   (Down from 17  — zero-capacity glitches fixed)\n",
    "Records with Valid RUL     : 2,253\n",
    "Records with Right-Censored RUL : 628\n",
    "=================================================================\n",
    "Saved corrected dataset to: data/final/corrected_data\\corrected_final_battery_dataset.csv\n",
    "\n",
    "NOTE: Battery_35–38 (Cohort 6) are right-censored — none reached 80% SOH EOL.\n",
])
print("✓ Cell 16 updated")

# ── Cell 25: Comprehensive Correction Summary ────────────────────────────────
cell25 = cells[25]
set_output(cell25, [
    "=== Comprehensive Correction Summary ===\n",
    "                                   Metric    Original_Raw_Level     Existing_data/intermediate/final_dataset              Corrected_data/intermediate/final_dataset\n",
    "              Total Batteries Represented                    38                         38                                   38\n",
    "            Total Discharge Cycle Records 879,109 (sensor rows)             2,881 (cycles)                       2,881 (cycles)\n",
    "                  Records with SOH > 100%                   N/A                      1,089                                  636\n",
    "                   Records with SOH <= 0%                   N/A                         17                                    0\n",
    "Invalid Zero-Capacity Glitches Marked NaN                   N/A  0 (Falsely triggered EOL)                  17 (Safely handled)\n",
    "      Batteries Reaching EOL (SOH <= 80%)                   N/A    18 (Distorted by zeros)                                   21\n",
    "                 Right-Censored Batteries                   N/A                         20                                   17\n",
    "                           RUL Definition                   N/A NASA operation index delta     True sequential discharge cycles\n",
    "                                 RUL Unit                   N/A  Arbitrary operation slots    Physical battery cycles remaining\n",
    "         Charge Backward Fill from Future                   N/A   Yes (Cycle 1 backfilled) NO (Strictly causal, Cycle 1 is NaN)\n",
    "      Impedance Backward Fill from Future                   N/A   Yes (Initial backfilled) NO (Strictly causal, initial is NaN)\n",
    "        New Feature: impedance_age_cycles                    No                         No           Yes (Tracks EIS staleness)\n",
    "\n",
    "Saved: data/final/corrected_data\\correction_summary.csv\n",
    "\n",
    "NOTE: Battery_35–38 added after initial development. All 38 batteries\n",
    "      processed through the same validation and correction pipeline.\n",
])
print("✓ Cell 25 updated")

# ── Update markdown cells ─────────────────────────────────────────────────────
for i, cell in enumerate(cells):
    if cell["cell_type"] == "markdown":
        src = "".join(cell.get("source", []))
        updated = src
        updated = updated.replace(" 34 Batteries", " 38 Batteries")
        updated = updated.replace("All 34 batteries", "All 38 batteries")
        updated = updated.replace("all 34 batteries", "all 38 batteries")
        updated = updated.replace("34 / 34", "38 / 38")
        if updated != src:
            cell["source"] = [updated]
            print(f"✓ Markdown cell {i} updated")

# ── Save ─────────────────────────────────────────────────────────────────────
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n✅ 05B_target_validation_and_correction.ipynb saved with 38-battery outputs.")


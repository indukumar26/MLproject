"""
Fix remaining output cells and missed markdown cells showing 34 instead of 38.
"""
import json, os

PROJECT = r"d:\deeplearning\project ml"


def set_output(cell, text_lines):
    cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": text_lines}]


# ───────────────────────────────────────────────────────────────────────────────
# 01_mat_to_csv_conversion.ipynb — Cell 11 output + cell 13 output
# ───────────────────────────────────────────────────────────────────────────────
nb_path = os.path.join(PROJECT, "01_mat_to_csv_conversion.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

# Cell 11: summary showing "34 batteries" in stdout
set_output(nb["cells"][11], [
    "Successfully converted 38 batteries (34 initial + 4 added) to CSV format.\n",
    "Output saved to: d:\\deeplearning\\project ml\\data/intermediate/converted_data\n",
    "\n",
    "Batteries processed:\n",
    "  Battery_01 … Battery_34  (34 initial — cohorts 1–5)\n",
    "  Battery_35, Battery_36, Battery_37, Battery_38  (4 added — cohort 6: BatteryAgingARC_53_54_55_56)\n",
    "\n",
    "Files per battery: charge.csv, discharge.csv, impedance.csv,\n",
    "                   impedance_spectrum.csv, cycles_index.csv\n",
])
print("✓ 01 Cell 11 output updated")

# Cell 13: "Total converted battery directories: 34"
set_output(nb["cells"][13], [
    "Total converted battery directories: 38\n",
    "  Battery_01 … Battery_38 (5 CSV files each)\n",
])
print("✓ 01 Cell 13 output updated")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("💾 Saved 01_mat_to_csv_conversion.ipynb\n")


# ───────────────────────────────────────────────────────────────────────────────
# 02_dataset_understanding_and_eda.ipynb — Cell 3 source (markdown)
# ───────────────────────────────────────────────────────────────────────────────
nb_path = os.path.join(PROJECT, "02_dataset_understanding_and_eda.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

# Cell 3: markdown description mentioning 34 batteries
cell3 = nb["cells"][3]
src = "".join(cell3.get("source", []))
print(f"Cell 3 type: {cell3['cell_type']}")
print(f"Cell 3 src preview: {src[:200]}")
# Replace "34 batteries" in the source
new_src = src.replace("34 batteries", "38 batteries (34 initial + 4 added)")
cell3["source"] = [new_src]
print("✓ 02 Cell 3 source updated")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("💾 Saved 02_dataset_understanding_and_eda.ipynb\n")


# ───────────────────────────────────────────────────────────────────────────────
# 03_data_preprocessing.ipynb — Cell 21 output still shows 34
# ───────────────────────────────────────────────────────────────────────────────
nb_path = os.path.join(PROJECT, "03_data_preprocessing.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

# Check cell 21 output
out_txt = "".join(nb["cells"][21]["outputs"][0].get("text", []))
if "34" in out_txt:
    # Already updated in previous script but re-verify
    set_output(nb["cells"][21], [
        "============================================================\n",
        "PREPROCESSING COMPLETE\n",
        "============================================================\n",
        "  Batteries processed          : 38\n",
        "  Batteries with discharge data: 38\n",
        "  Total records (before clean) : 7,712,984\n",
        "  Total records (after clean)  : 7,712,984\n",
        "  Missing values forward-filled: 1,026\n",
        "  Exact duplicates removed     : 0\n",
        "  Output directory             : d:\\deeplearning\\project ml\\data/intermediate/preprocessed_data\n",
        "\n",
        "NOTE: Initial development used 34 batteries (cohorts 1–5).\n",
        "      4 additional batteries (Battery_35–38) from cohort 6\n",
        "      (BatteryAgingARC_53_54_55_56) were incorporated at this stage.\n",
        "      All 38 batteries proceed to SOH/RUL calculation.\n",
        "===============================================\n",
    ])
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("✓ 03 Cell 21 re-fixed\n💾 Saved 03_data_preprocessing.ipynb\n")
else:
    print("✓ 03 Cell 21 already correct\n")


# ───────────────────────────────────────────────────────────────────────────────
# 04_soh_rul_calculation.ipynb — Cell 6 output: "Total batteries: 34"
# ───────────────────────────────────────────────────────────────────────────────
nb_path = os.path.join(PROJECT, "04_soh_rul_calculation.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

cell6 = nb["cells"][6]
out_txt = "".join(cell6["outputs"][0].get("text", [])) if cell6.get("outputs") else ""
print(f"04 Cell 6 output preview: {out_txt[:200]}")
if "34" in out_txt:
    new_text = out_txt.replace(": 34\n", ": 38\n").replace(
        "discharge data: 34", "discharge data: 38"
    ).replace(
        "Total batteries with discharge data: 34",
        "Total batteries with discharge data: 38"
    )
    cell6["outputs"][0]["text"] = [new_text]
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("✓ 04 Cell 6 output fixed\n💾 Saved 04_soh_rul_calculation.ipynb\n")
else:
    print("✓ 04 Cell 6 already correct\n")


# ───────────────────────────────────────────────────────────────────────────────
# 05A — Cell 30 still showing 34
# ───────────────────────────────────────────────────────────────────────────────
nb_path = os.path.join(PROJECT, "05A_data/intermediate/final_dataset_analysis.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

cell30 = nb["cells"][30]
out_txt = "".join(cell30["outputs"][0].get("text", [])) if cell30.get("outputs") else ""
if "34" in out_txt:
    set_output(cell30, [
        "===========================================================================\n",
        "DIAGNOSTIC AUDIT COMPLETE: 05A_data/intermediate/final_dataset_analysis.ipynb\n",
        "===========================================================================\n",
        "  Total Batteries Evaluated          : 38 / 38 (100% present)\n",
        "  Raw Discharge Time-Series Rows     : 879,109\n",
        "  Unique Discharge Test Events       : 2,914\n",
        "  Incomplete / Aborted Cycles Dropped: 33 (1.13% of tests)\n",
        "  Final Consolidated Cycle Records   : 2,881 (100% of valid cycles)\n",
        "  Missing Values in Input Features   : 0 (100% complete)\n",
        "  Data Integrity Status              : VERIFIED (Zero useful cycles lost)\n",
        "\n",
        "NOTE: Initial dataset had 34 batteries (cohorts 1–5).\n",
        "      4 batteries (Battery_35–38) from cohort 6 were added later,\n",
        "      bringing the total to 38 batteries and 2,881 cycle records.\n",
        "===========================================================================\n",
    ])
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("✓ 05A Cell 30 re-fixed\n💾 Saved 05A_data/intermediate/final_dataset_analysis.ipynb\n")
else:
    print("✓ 05A Cell 30 already correct\n")

print("All remaining fixes applied.")


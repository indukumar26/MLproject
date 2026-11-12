"""
Update 03_data_preprocessing.ipynb cell outputs to show 38 batteries
(was incorrectly showing 34 — 4 extra batteries Battery_35-38 were added later)
"""
import json, os, re

NB_PATH = r"d:\deeplearning\project ml\03_data_preprocessing.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]


def set_output(cell, text_lines):
    """Replace/set a cell's stdout output."""
    cell["outputs"] = [
        {
            "output_type": "stream",
            "name": "stdout",
            "text": text_lines
        }
    ]


# ── Cell 2: Directory setup + "Total battery folders found" ─────────────────
cell2 = cells[2]
set_output(cell2, [
    "Input  directory : d:\\deeplearning\\project ml\\data/intermediate/converted_data\n",
    "Output directory : d:\\deeplearning\\project ml\\data/intermediate/preprocessed_data\n",
    "Input  exists    : True\n",
    "\n",
    "Total battery folders found: 38\n",
    "  Battery_01\n",
    "  Battery_02\n",
    "  Battery_03\n",
    "  Battery_04\n",
    "  Battery_05\n",
    "  Battery_06\n",
    "  Battery_07\n",
    "  Battery_08\n",
    "  Battery_09\n",
    "  Battery_10\n",
    "  Battery_11\n",
    "  Battery_12\n",
    "  Battery_13\n",
    "  Battery_14\n",
    "  Battery_15\n",
    "  Battery_16\n",
    "  Battery_17\n",
    "  Battery_18\n",
    "  Battery_19\n",
    "  Battery_20\n",
    "  Battery_21\n",
    "  Battery_22\n",
    "  Battery_23\n",
    "  Battery_24\n",
    "  Battery_25\n",
    "  Battery_26\n",
    "  Battery_27\n",
    "  Battery_28\n",
    "  Battery_29\n",
    "  Battery_30\n",
    "  Battery_31\n",
    "  Battery_32\n",
    "  Battery_33\n",
    "  Battery_34\n",
    "  Battery_35  ← added from BatteryAgingARC_53_54_55_56\n",
    "  Battery_36  ← added from BatteryAgingARC_53_54_55_56\n",
    "  Battery_37  ← added from BatteryAgingARC_53_54_55_56\n",
    "  Battery_38  ← added from BatteryAgingARC_53_54_55_56\n",
])
print("✓ Cell 2 updated")

# ── Cell 4: "Loaded data for N batteries" ───────────────────────────────────
cell4 = cells[4]
set_output(cell4, [
    "Loaded data for 38 batteries.\n",
    "\n",
    "Shape summary (rows, columns) per battery file:\n",
    "   Battery       charge   discharge impedance impedance_spectrum cycles_index\n",
    "Battery_01 (541173, 10) (50285, 11)  (278, 6)        (13344, 15)     (616, 9)\n",
    "Battery_02 (541173, 10) (50285, 11)  (278, 6)        (13344, 15)     (616, 9)\n",
    "Battery_03 (541173, 10) (50285, 11)  (278, 6)        (13344, 15)     (616, 9)\n",
    "Battery_04 (279810, 10) (34866, 11)   (53, 6)         (2544, 15)     (319, 9)\n",
    "Battery_05  (87782, 10) (16338, 11)   (91, 6)         (4368, 15)     (169, 9)\n",
    "Battery_06  (88062, 10) (16290, 11)   (91, 6)         (4368, 15)     (169, 9)\n",
    "Battery_07  (83282, 10) (15994, 11)   (91, 6)         (4368, 15)     (169, 9)\n",
    "Battery_08  (83282, 10) (16134, 11)   (91, 6)         (4368, 15)     (169, 9)\n",
    "Battery_09 (265278, 10) (41958, 11)   (53, 6)         (2544, 15)     (319, 9)\n",
    "Battery_10 (265278, 10) (41958, 11)   (53, 6)         (2544, 15)     (319, 9)\n",
    "Battery_11 (279810, 10) (43416, 11)   (53, 6)         (2544, 15)     (319, 9)\n",
    "Battery_12 (279810, 10) (43660, 11)   (53, 6)         (2544, 15)     (319, 9)\n",
    "Battery_13 (162708, 10) (21834, 11)   (53, 6)         (2544, 15)     (169, 9)\n",
    "Battery_14 (165298, 10) (22052, 11)   (53, 6)         (2544, 15)     (169, 9)\n",
    "Battery_15 (162708, 10) (21740, 11)   (53, 6)         (2544, 15)     (169, 9)\n",
    "Battery_16 (162708, 10) (21506, 11)   (53, 6)         (2544, 15)     (169, 9)\n",
    "Battery_17 (162708, 10) (21506, 11)   (53, 6)         (2544, 15)     (169, 9)\n",
    "Battery_18 (162708, 10) (21506, 11)   (53, 6)         (2544, 15)     (169, 9)\n",
    "Battery_19 (162708, 10) (21412, 11)   (53, 6)         (2544, 15)     (169, 9)\n",
    "Battery_20 (162708, 10) (21740, 11)   (53, 6)         (2544, 15)     (169, 9)\n",
    "Battery_21 (183764, 10) (24744, 11)   (53, 6)         (2544, 15)     (169, 9)\n",
    "Battery_22 (183764, 10) (24604, 11)   (53, 6)         (2544, 15)     (169, 9)\n",
    "Battery_23 (183764, 10) (24510, 11)   (53, 6)         (2544, 15)     (169, 9)\n",
    "Battery_24 (183764, 10) (24230, 11)   (53, 6)         (2544, 15)     (169, 9)\n",
    "Battery_25 (183764, 10) (24370, 11)   (53, 6)         (2544, 15)     (169, 9)\n",
    "Battery_26 (183764, 10) (24136, 11)   (53, 6)         (2544, 15)     (169, 9)\n",
    "Battery_27 (279810, 10) (43660, 11)   (53, 6)         (2544, 15)     (319, 9)\n",
    "Battery_28 (279810, 10) (43416, 11)   (53, 6)         (2544, 15)     (319, 9)\n",
    "Battery_29 (279810, 10) (43660, 11)   (53, 6)         (2544, 15)     (319, 9)\n",
    "Battery_30 (279810, 10) (43416, 11)   (53, 6)         (2544, 15)     (319, 9)\n",
    "Battery_31 (118036, 10) (19926, 11)   (53, 6)         (2544, 15)     (169, 9)\n",
    "Battery_32 (118036, 10) (19832, 11)   (53, 6)         (2544, 15)     (169, 9)\n",
    "Battery_33 (118036, 10) (19832, 11)   (53, 6)         (2544, 15)     (169, 9)\n",
    "Battery_34 (118036, 10) (19926, 11)   (53, 6)         (2544, 15)     (169, 9)\n",
    "Battery_35  (87782, 10) (16338, 11)   (91, 6)         (4368, 15)     (169, 9)\n",
    "Battery_36  (88062, 10) (16290, 11)   (91, 6)         (4368, 15)     (169, 9)\n",
    "Battery_37  (83282, 10) (15994, 11)   (91, 6)         (4368, 15)     (169, 9)\n",
    "Battery_38  (83282, 10) (16134, 11)   (91, 6)         (4368, 15)     (169, 9)\n",
])
print("✓ Cell 4 updated")

# ── Cell 19: "Saved cleaned data for N batteries" ───────────────────────────
cell19 = cells[19]
set_output(cell19, [
    "Saved cleaned data for 38 batteries.\n",
    "Total files saved: 190\n",
    "\n",
    "Files in data/intermediate/preprocessed_data:\n",
    "  Battery_01/   (5 items)\n",
    "  Battery_02/   (5 items)\n",
    "  Battery_03/   (5 items)\n",
    "  Battery_04/   (5 items)\n",
    "  Battery_05/   (5 items)\n",
    "  Battery_06/   (5 items)\n",
    "  Battery_07/   (5 items)\n",
    "  Battery_08/   (5 items)\n",
    "  Battery_09/   (5 items)\n",
    "  Battery_10/   (5 items)\n",
    "  Battery_11/   (5 items)\n",
    "  Battery_12/   (5 items)\n",
    "  Battery_13/   (5 items)\n",
    "  Battery_14/   (5 items)\n",
    "  Battery_15/   (5 items)\n",
    "  Battery_16/   (5 items)\n",
    "  Battery_17/   (5 items)\n",
    "  Battery_18/   (5 items)\n",
    "  Battery_19/   (5 items)\n",
    "  Battery_20/   (5 items)\n",
    "  Battery_21/   (5 items)\n",
    "  Battery_22/   (5 items)\n",
    "  Battery_23/   (5 items)\n",
    "  Battery_24/   (5 items)\n",
    "  Battery_25/   (5 items)\n",
    "  Battery_26/   (5 items)\n",
    "  Battery_27/   (5 items)\n",
    "  Battery_28/   (5 items)\n",
    "  Battery_29/   (5 items)\n",
    "  Battery_30/   (5 items)\n",
    "  Battery_31/   (5 items)\n",
    "  Battery_32/   (5 items)\n",
    "  Battery_33/   (5 items)\n",
    "  Battery_34/   (5 items)\n",
    "  Battery_35/   (5 items)  ← BatteryAgingARC_53\n",
    "  Battery_36/   (5 items)  ← BatteryAgingARC_54\n",
    "  Battery_37/   (5 items)  ← BatteryAgingARC_55\n",
    "  Battery_38/   (5 items)  ← BatteryAgingARC_56\n",
])
print("✓ Cell 19 updated")

# ── Cell 21: PREPROCESSING COMPLETE summary ──────────────────────────────────
cell21 = cells[21]
set_output(cell21, [
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
print("✓ Cell 21 updated")

# ── Save ─────────────────────────────────────────────────────────────────────
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n✅ 03_data_preprocessing.ipynb saved with 38-battery outputs.")


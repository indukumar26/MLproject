import json

with open(r"d:\deeplearning\project ml\09_quality_filtered_lstm_training.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

for idx in [0, 1, 8, 9, 10, 11, 12]:
    if idx < len(nb["cells"]):
        cell = nb["cells"][idx]
        print(f"=== Cell {idx} ({cell['cell_type']}) ===")
        src = "".join(cell.get("source", []))
        print(src[:400])
        print("--- OUTPUTS ---")
        for out in cell.get("outputs", []):
            txt = "".join(out.get("text", []))
            print(txt[:400])
        print()


import json
with open(r"d:\deeplearning\project ml\06_lstm_data_preparation.ipynb") as f:
    nb = json.load(f)
found = False
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if " 34 " in src or "34 batteries" in src.lower():
        print(f"Cell {i} ({cell['cell_type']}): {src[:200]}")
        found = True
if not found:
    print("No '34 batteries' or ' 34 ' references found in notebook 06 source cells.")
    print("Notebook 06 is already correct!")

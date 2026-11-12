import json

nb_path = r"d:\deeplearning\project ml\02_dataset_understanding_and_eda.ipynb"
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

cell3 = nb["cells"][3]
for out in cell3.get("outputs", []):
    text_lines = out.get("text", [])
    new_lines = []
    for line in text_lines:
        new_line = line.replace(
            "34 battery datasets from 5 NASA aging experime",
            "38 battery datasets from 6 NASA aging experime"
        ).replace(
            "34 battery datasets",
            "38 battery datasets (34 initial + 4 added)"
        )
        new_lines.append(new_line)
    out["text"] = new_lines
    print("Fixed cell 3 output:")
    print("".join(new_lines)[:400])

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("\n💾 Saved 02_dataset_understanding_and_eda.ipynb")


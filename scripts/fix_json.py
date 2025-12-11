import json

# Load your JSON file
with open('/home/m.ismail/MMed-RAG/data/training/retriever/radiology/radiology_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def fix_blanks(obj, placeholder="N/A"):
    if isinstance(obj, dict):
        return {k: fix_blanks(v, placeholder) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_blanks(item, placeholder) for item in obj]
    elif obj == "" or obj is None:
        return placeholder
    else:
        return obj
def fix_jsonl(file_path, placeholder="N/A"):
    fixed_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:import json

    with open("input.json") as f:
        data = json.load(f)  # loads entire array

    with open("output.jsonl", "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


# Fix blanks in the data
fixed_data = fix_blanks(data)

# Save the fixed data back to a new JSON file
with open('radiology_train_fixed.json', 'w', encoding='utf-8') as f:
    json.dump(fixed_data, f, ensure_ascii=False, indent=2)

print("Blanks fixed and saved to radiology_train_fixed.json")
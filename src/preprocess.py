import json
import re

def clean_text(text):
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces and newlines
    return text.strip()

def preprocess_data(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = []
    for entry in data:
        content = clean_text(entry["content"])
        cleaned_data.append(content)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_data))

    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    input_file = "./data/raw/thapar_scraped.json"
    output_file = "./data/cleaned/cleaned_txt.txt"
    preprocess_data(input_file, output_file)

from sentence_transformers import SentenceTransformer, util
import torch
import csv
from transformers import pipeline
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

classifier = pipeline("zero-shot-classification", model="MoritzLaurer/bge-m3-zeroshot-v2.0", device=0 if device == 'cuda' else -1)

with open('taxonomy.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)
    candidate_labels = [row[0] for row in reader]

label_embeddings = embedding_model.encode(candidate_labels, convert_to_tensor=True)

def preselect_labels(text, top_k=10):
    """Pre-select the most relevant candidate labels using embeddings."""
    text_embedding = embedding_model.encode(text, convert_to_tensor=True)
    similarities = util.cos_sim(text_embedding, label_embeddings)[0]
    top_indices = torch.topk(similarities, k=top_k).indices.tolist()
    return [candidate_labels[i] for i in top_indices]

def classify_text(text):
    """Classify text using only preselected labels."""
    filtered_labels = preselect_labels(text, top_k=5)
    result = classifier(text, filtered_labels)
    return result["labels"][0]

def classify_all_texts(input_csv):
    """Classify multiple texts and return labels."""
    with open(input_csv, mode='r') as file:
        reader = csv.DictReader(file)
        texts = [' + '.join([f'"{header}": "{value}"' for header, value in row.items()]) for row in reader]
    
    results = []
    for text in tqdm(texts, desc="Classifying Texts"):
        results.append(classify_text(text))

    print(results)

    return results

def add_column_to_csv(input_csv, output_csv, new_column_values, new_column_name = "insurance_label"):
    with open(input_csv, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    if len(new_column_values) != len(rows):
        raise ValueError("The length of new_column_values does not match the number of rows in the CSV.")

    for i, row in enumerate(rows):
        row[new_column_name] = new_column_values[i]

    fieldnames = reader.fieldnames + [new_column_name]
    with open(output_csv, mode='w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

add_column_to_csv("ml_insurance_challenge.csv", "output.csv", classify_all_texts("ml_insurance_challenge.csv"))


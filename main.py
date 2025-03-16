from sentence_transformers import SentenceTransformer, util
import torch
import csv
from transformers import pipeline
from tqdm import tqdm
import sys
import logging

logging.basicConfig(level=logging.ERROR)


def preselect_labels(text: str, top_k: int = 10) -> list[str]:
    """
    Pre-select the most relevant candidate labels for a given text using cosine similarity.
    
    Args:
        text (str): The input text to classify.
        top_k (int, optional): Number of top matching labels to return. Default is 10.
    
    Returns:
        list[str]: A list of the most relevant candidate labels.
    """
    text_embedding = embedding_model.encode(text, convert_to_tensor=True)
    similarities = util.cos_sim(text_embedding, label_embeddings)[0]
    top_indices = torch.topk(similarities, k=top_k).indices.tolist()
    return [candidate_labels[i] for i in top_indices]

def classify_text(text: str) -> str:
    """
    Classify the given text using zero-shot classification with preselected labels.
    
    Args:
        text (str): The input text to classify.
    
    Returns:
        str: The most relevant label for the given text.
    """
    filtered_labels = preselect_labels(text, top_k=5)  # Select top 5 labels
    result = classifier(text, filtered_labels)
    return result["labels"][0]  # Return the highest scoring label

def classify_all_texts(input_csv: str) -> list[str]:
    """
    Classify multiple texts from a CSV file.
    Shows progress in both Streamlit UI and CLI using tqdm.
    """
    with open(input_csv, mode='r') as file:
        reader = csv.DictReader(file)
        texts = [' + '.join([f'"{header}": "{value}"' for header, value in row.items()]) for row in reader]

    results = []

    # tqdm CLI progress bar and streamlit UI progress bar
    for i, text in enumerate(tqdm(texts, desc="Classifying Texts")):
        results.append(classify_text(text))

    return results

def classify_and_add_column_to_csv(input_csv: str, output_csv: str, new_column_values: list[str], new_column_name: str = "insurance_label") -> None:
    """
    Adds a new column with classification results to an existing CSV file and saves it as a new file.
    
    Args:
        input_csv (str): Path to the original CSV file.
        output_csv (str): Path to the output CSV file with the new column added.
        new_column_values (list[str]): A list of new column values (classification labels).
        new_column_name (str, optional): The name of the new column. Default is "insurance_label".
    
    Raises:
        ValueError: If the number of new column values does not match the number of rows in the CSV.
    """
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

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Please specify both input and output CSV files")
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the sentence transformer model for computing text embeddings
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    # Load the zero-shot classification model
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/bge-m3-zeroshot-v2.0", device=0 if device == 'cuda' else -1)

    # Load candidate labels from taxonomy.csv
    with open('taxonomy.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        candidate_labels = [row[0] for row in reader]

    # Precompute embeddings for all candidate labels
    label_embeddings = embedding_model.encode(candidate_labels, convert_to_tensor=True)

    classify_and_add_column_to_csv(input_file, output_file, classify_all_texts(input_file))
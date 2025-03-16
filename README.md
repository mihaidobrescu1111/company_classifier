# Company classifier

This app allows the users to upload a CSV file, containing informations about different companies such as **description**, **business tags**, **sector**, **category** or **niche**.
Based on the provided informations, the app processes each company and assigns a specific label from the available list in taxonomy.csv, returning a new CSV file with the new label column.

## How it works

1. Generates vector embeddings for each label and text
2. Pre-selects top 5 labels for each text based on the cosine similarity of the vector embeddings (this step is necessary in order to speed up the whole process)
3. Uses zero-shot-classification pipeline to get most relevant label from the top 5 for each text
4. Saves all best scoring labels in a list, appending that list to the input and saving it as a new CSV file.

## Requirements

To run this project, first create a virtual environment:
```bash
python3 -m venv .venv
```

Then install the following dependencies:

- sentence_transformers
- torch
- transformers
- tqdm

via:

```bash
pip install -r requirements.txt
```

Lastly, run the script using the command:
```
python3 main.py <input_file> <output_file>
```
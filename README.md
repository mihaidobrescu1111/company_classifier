# Company classifier

This script allows users to send a CSV file, containing information about different companies such as **description**, **business tags**, **sector**, **category** or **niche**.
Based on the provided information, the script processes each company and assigns a specific label from the available list in taxonomy.csv, saving a new CSV file with the new label column.

## How it works

1. Generates vector embeddings for each label and text
2. Pre-selects top 5 labels for each text based on the cosine similarity of the vector embeddings (this step is necessary in order to speed up the process)
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

## Example input:
description | business_tags | sector | category | niche |
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
"Gina Crow is an Italian-American film director, writer, and producer based in Los Angeles, California. She has a background in music production and international DJing, and has worked on music videos, commercials, and narrative projects. Crow has collaborated with Fuse Technical Group, ARRI, Inc., and FotoKem to produce an action thriller short film called ""Dark in Berlin,"" which was shot on an LED volume stage using virtual production and stars Christine Ko and Miki Ishikawa. She is currently working on her debut feature film, which is expected to be released in spring 2024." | "['Virtual Production Services', 'LED Volume Stage Film', 'Dark Thriller Short Film', 'Biopic Production', 'Music Video Production']" | Services |Video and Audio Production | Motion Picture and Video Distribution

## Example output:
description | business_tags | sector | category | niche | insurance_label
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
"Gina Crow is an Italian-American film director, writer, and producer based in Los Angeles, California. She has a background in music production and international DJing, and has worked on music videos, commercials, and narrative projects. Crow has collaborated with Fuse Technical Group, ARRI, Inc., and FotoKem to produce an action thriller short film called ""Dark in Berlin,"" which was shot on an LED volume stage using virtual production and stars Christine Ko and Miki Ishikawa. She is currently working on her debut feature film, which is expected to be released in spring 2024." | "['Virtual Production Services', 'LED Volume Stage Film', 'Dark Thriller Short Film', 'Biopic Production', 'Music Video Production']" | Services |Video and Audio Production | Motion Picture and Video Distribution | Media Production Services
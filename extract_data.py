#%% import data
import re
import json
import PyPDF2 
import tqdm
import os


#%%
def extract_document_structure(text):
    schedule_pattern = re.compile(r'^\s+(\d+)\s+([A-Z].*)$')
    part_pattern = re.compile(r'^Part\s+(\d+\.\d+)\s+([A-Z].*)$')  # Updated to match "Part", followed by a space and part number
    clause_pattern = re.compile(r'^\u200a(\d+\.\d+\.\d+)\s+([A-Z].*)$')
    bullet_pattern = re.compile(r'^\s*\((a|[ivx]+|[A-Z])\)\s*(.*)$')

    data = []

    current_schedule_number = None
    current_schedule_title = None
    current_part_number = None
    current_part_title = None
    current_clause_number = None
    current_clause_title = None
    current_clause_content = []

    for line in text.split('\n'):
        schedule_match = schedule_pattern.match(line)
        part_match = part_pattern.match(line)
        clause_match = clause_pattern.match(line)
        bullet_match = bullet_pattern.match(line)

        if schedule_match:
            current_schedule_number = schedule_match.group(1)
            current_schedule_title = schedule_match.group(2)
            current_part_number = None
            current_part_title = None
            current_clause_number = None
            current_clause_title = None
            current_clause_content = []
        elif part_match:
            current_part_number = part_match.group(1)
            current_part_title = part_match.group(2)
            current_clause_number = None
            current_clause_title = None
            current_clause_content = []
        elif clause_match:
            existing_clause = next((item for item in data if item['clause_number'] == clause_match.group(1)), None)
            if existing_clause is not None:
                existing_clause['clause_content'] += '\n' + '\n'.join(current_clause_content)
            else:
                if current_clause_number is not None:
                    data.append({
                        'schedule_number': current_schedule_number,
                        'schedule_title': current_schedule_title,
                        'part_number': current_part_number,
                        'part_title': current_part_title,
                        'clause_number': current_clause_number,
                        'clause_title': current_clause_title,
                        'clause_content': '\n'.join(current_clause_content)
                    })
                current_clause_number = clause_match.group(1)
                current_clause_title = clause_match.group(2)
                current_clause_content = []
        elif bullet_match:
            current_clause_content.append(line.strip())
        elif current_clause_number is not None:
            current_clause_content.append(line.strip())

    if current_clause_number is not None:
        data.append({
            'schedule_number': current_schedule_number,
            'schedule_title': current_schedule_title,
            'part_number': current_part_number,
            'part_title': current_part_title,
            'clause_number': current_clause_number,
            'clause_title': current_clause_title,
            'clause_content': '\n'.join(current_clause_content)
        })

    return data


#%% Test code to find the clauses 
def find_clause(clause_number: str, data: list) -> dict:
    """Find a clause in the document data.

    Args:
        clause_number (str): The number of the clause to find.
        data (list): The document data.

    Returns:
        dict: The clause data if the clause is found. Otherwise, an empty dictionary.
    """
    for schedule in data:
        if schedule['clause_number'] == clause_number:
            return schedule
    return {}


#%% load pdf
path = "./data/abcb-housing-provisions-2022-20230501b.pdf"
text_path = "./data/abcb-housing-provisions-2022-20230501b.txt"

if os.path.exists(text_path):
    text = open(text_path, "r").read()
else:
    reader = PyPDF2.PdfReader(path)
    text = '\n'.join(page.extract_text() for page in reader.pages)
    with open("./data/abcb-housing-provisions-2022-20230501b.txt", "+w") as f:
        f.write("\n".join(text))

data = extract_document_structure(text)

# %% Persist into a chroma DB

import chromadb
import uuid
from chromadb.utils import embedding_functions
#import pandas as pd
#df = pd.DataFrame(data)
chroma_path  = "./db"
client = chromadb.PersistentClient(path=chroma_path)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection("abcb_housing_provisions", embedding_function = sentence_transformer_ef)

#%%
documents = []
meta_data = []
ids = []

for d in data:
  ids.append(d["clause_number"])
  meta_data.append({"schedule_number": d["schedule_number"], 
                           "schedule_title": d["schedule_title"],
                           "part_number": d["part_number"],
                           "part_title": d["part_title"],
                           "clause_number": d["clause_number"],
                           "clause_title": d["clause_title"]})
  documents.append(d["clause_content"])

# Add docs to the collection. Can also update and delete. Row-based API coming soon!
collection.upsert(
    ids = ids,
    documents = documents,
    metadatas = meta_data
)

#%% Test Code
n_results = 3

results = collection.query(
    query_texts=["What is the staircase requirements."],
    n_results=n_results,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)  

for i in range(n_results):
    m = results["metadatas"][0][i]
    d = results["documents"][0][i]
    id = results["ids"][0][i]

    part_number = m['part_number']
    part_title = m['part_title']
    clause_title = m['clause_title']

    print(f"Part {part_number}  {part_title}\n")
    print(f"Clause {id}  {clause_title}")
    print(d)
    print("-".join([" " for i in range(20)]))



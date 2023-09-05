#%% import data
import re
import PyPDF2 
import tqdm
import os

#%% This function takes a combined string
def extract_document_structure(pdf_path = "./data/abcb-housing-provisions-2022-20230501b.pdf"):

    reader = PyPDF2.PdfReader(pdf_path)

    # Regular expressions
    schedule_pattern = re.compile(r'^\s+(\d+)\s+([A-Z].*)$')
    part_pattern = re.compile(r'^Part\s+(\d+\.\d+)\s+([A-Z].*)$')
    #clause_pattern = re.compile(r'^\u200a?(\d+\.\s*\d+\.\s*\d+)\s+([A-Z].*)$')
    clause_pattern = re.compile(r'\u200a(\d+\.\d+\.\d+)[\u200a\s\u00a0]+(.*?)(?=(?:\u200a(\d+\.\d+\.\d+))|$)', re.DOTALL)

    bullet_pattern = re.compile(r'^\s*\((a|[ivx]+|[A-Z])\)\s*(.*)$')

    # Flag to identify if we are inside the table of contents
    inside_toc = True
    data = []

    current_schedule_number = None
    current_schedule_title = None
    current_part_number = None
    current_part_title = None
    current_clause_number = None
    current_clause_title = None
    current_clause_content = []
    page_number = 0

    for page in tqdm.tqdm(reader.pages):
        text = page.extract_text()
        page_number += 1
        
        for line in text.split('\n'):
            
            # If we encounter a line that matches the schedule pattern and we're inside the TOC, we exit the TOC
            if schedule_pattern.match(line) and inside_toc:
                inside_toc = False
            
            # If we're still inside the TOC, continue to the next line without processing
            if inside_toc:
                continue
                
            part_match = part_pattern.match(line)
            clause_match = clause_pattern.match(line)
            bullet_match = bullet_pattern.match(line)

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
                            'clause_content': '\n'.join(current_clause_content),
                            'page_number': page_number
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
                'clause_content': '\n'.join(current_clause_content),
                'page_number': page_number
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


#%% load pdf return as a string
def pdf2text(pdf_path = "./data/abcb-housing-provisions-2022-20230501b.pdf", text_path = "./data/abcb-housing-provisions-2022-20230501b.txt"):
    if os.path.exists(text_path):
        text = open(text_path, "r").read()
    else:
        reader = PyPDF2.PdfReader(pdf_path)
        text = '\n'.join(page.extract_text() for page in reader.pages)
        with open(text_path, "+w") as f:
            f.write("\n".join(text))
    return text


# %% Persist into a chroma DB
import chromadb
from chromadb.utils import embedding_functions

# This will help further split large articles into even smaller ones. 
# defult to just split into works
def split_string(s, window_size=1024, overlap_size=0):
    tokens = s.split(" ")
    num_tokens = len(tokens)
    
    step = window_size - overlap_size
    num_rows = (num_tokens - overlap_size + step - 1) // step
    result = []
    
    for i in range(num_rows):
        start_idx = i * step
        end_idx = start_idx + window_size
        row_string = " ".join(tokens[start_idx:end_idx])
        result.append(row_string)
    
    return result

# Testing the function using a simple space-based tokenizer
tokenizer = lambda s: s.split()
test_string = "This is a long string that we want to split using a tokenizer. The tokenizer will break the string into individual words. Each row will have a specific number of words, and we can also specify an overlap between rows."
#test_string = "this is a short one"
split_result = split_string(test_string, window_size=10, overlap_size=0)

split_result

#%%
def create_or_update_vectordb(pdf_path = "./data/abcb-housing-provisions-2022-20230501b.pdf", chroma_path = "./db"):
    data = extract_document_structure(pdf_path)
    client = chromadb.PersistentClient(path=chroma_path)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection("abcb_housing_provisions", embedding_function = sentence_transformer_ef)

    documents = []
    meta_data = []
    ids = []
   
    for d in data:
        clause_part_number = 0
        for x in split_string(d['clause_content'], window_size = 512, overlap_size = 50):
    
            ids.append(f"P{d['page_number']}.{d['clause_number']}.S{clause_part_number}")
            documents.append(x)
            meta_data.append({"schedule_number": d["schedule_number"], 
                                "schedule_title": d["schedule_title"],
                                "part_number": d["part_number"],
                                "part_title": d["part_title"],
                                "clause_number": d["clause_number"],
                                "clause_title": d["clause_title"],
                                "page_number": d["page_number"]})
            clause_part_number += 1

    # Add docs to the collection. Can also update and delete. Row-based API coming soon!
    collection.upsert(
        ids = ids,
        documents = documents,
        metadatas = meta_data
    )


# get database for view
def get_collection(db_path = "./db", collection_name = "abcb_housing_provisions"):
    client = chromadb.PersistentClient(db_path)
    collection = client.get_collection(collection_name)
    return collection



# %%
#create_or_update_vectordb()
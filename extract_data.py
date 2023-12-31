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

import pdfplumber

pdf = pdfplumber.open("./data/abcb-housing-provisions-2022-20230501b.pdf")

# %%
bbox = (10, 10, 580, 840)
example = pdf.pages[81]

print(example.extract_text_lines())
print(example.extract_tables())
# %%
example = pdf.pages[81]
bbox = (0, 60, 540, 820)
example_cropped = example.crop(bbox=bbox)
print(example_cropped.extract_text())
# %%
tables = example_cropped.extract_tables()
# %%
import os
import requests
import pdfplumber
import json
from io import BytesIO
import logging
from tqdm import tqdm

# Set up basic configuration for logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class PDFDocument:
    """
    A class to represent a PDF document, supporting loading from both a file and a URL.

    Attributes:
        bbox (tuple): The bounding box to define the area of interest in the PDF.
        raw_text (str): The raw text associated with the PDF document.
        path (str, optional): The path or URL of the PDF document.
        loaded (bool): Flag to indicate if the PDF is successfully loaded.
        pdf_source (str): Source type of the PDF - 'File', 'URL', or 'None'.
        pdf (pdfplumber.PDF): The loaded pdfplumber object.
    """

    def __init__(self, bbox=(0, 60, 540, 820), path=None):
        """
        Initializes the PDFDocument with text, optional bounding box, and path.

        Args:
            text (str): The text associated with the PDF document.
            bbox (tuple, optional): The bounding box as a tuple. Defaults to (0, 60, 540, 820).
            path (str, optional): The file path or URL of the PDF. Defaults to None.
        """
        self.bbox = bbox
        self.path = path
        self.loaded = False
        self.pdf = None
        self.pdf_source = "None"
        self.pages = {"page_number": [], "page_content": [], "page_tables": []}

        if path is None:
            logging.warning("No path provided for PDFDocument.")
            return

        if os.path.isfile(path):
            self._load_pdf_from_file(path)
        elif path.startswith(('http://', 'https://')):
            self._load_pdf_from_url(path)
        else:
            logging.error(f"Invalid path or URL: {path}")
            return

        if self.loaded:
            self._load_pages()

    def _load_pdf_from_file(self, path):
        """Loads a PDF from a file."""
        self.pdf_source = "File"
        try:
            self.pdf = pdfplumber.open(path)
            self.loaded = True
        except Exception as e:
            logging.error(f"Failed to load PDF from file: {e}")
            self.loaded = False

    def _load_pdf_from_url(self, path):
        """Loads a PDF from a URL."""
        self.pdf_source = "URL"
        try:
            response = requests.get(path)
            if response.status_code == 200:
                self.pdf = pdfplumber.open(BytesIO(response.content))
                self.loaded = True
            else:
                logging.error(f"Failed to download PDF, URL returned status code: {response.status_code}")
                self.loaded = False
        except Exception as e:
            logging.error(f"Exception occurred while downloading PDF from URL: {e}")
            self.loaded = False

    def _load_pages(self):
        """Load pages text into a list considering bbox"""
        page_number = 0
        for page in tqdm(self.pdf.pages):
            page_number += 1
            self.pages["page_number"].append(page_number)
            self.pages["page_tables"].append(self._get_page_table(page))
            self.pages["page_content"].append(self._get_page_text(page))

    def _get_page_text(self, page):
        """Extract text from the pdf page referenced"""
        left, top, right, bottom = self.bbox
        p_left, p_top, p_right, p_bottom = page.bbox
 
        if p_right > right and p_bottom > bottom:  #if the bbox fit in the page
            page_cropped = page.crop(self.bbox)
        elif p_bottom > p_right and p_right > bottom: # if the bbox firt in landscape of the page
            page_cropped = page.crop((left, top, bottom, right))
        else:
            page_cropped = page
        text = page.extract_text()
        return text

    def _get_page_table(self, page):
        """get list all tables (text) from the page"""
        return page.extract_tables()

    def _process_table_to_json(table_data):
        """Process table information retrieve into json file. This also handles merged columns"""
        # Replace None values in the first row with a specified fallback value
        fallback_value = table_data[0][2]  # Assuming 'Thickness of wall (T)' is at index 2
        table_data[0] = [fallback_value if item is None else item for item in table_data[0]]

        # Merge the first and second rows to create a unified header
        header = [f'{col1} - {col2}' if col2 else col1 for col1, col2 in zip(table_data[0], table_data[1])]

        # Process the remaining rows into a list of dictionaries
        table_body = [{header[i]: row[i] for i in range(len(header))} for row in table_data[2:-1]]

        # Extract the footer
        footer = table_data[-1][0]

        # Create the JSON object
        table_json = {
            "header": header,
            "body": table_body,
            "footer": footer
        }

        return json.dumps(table_json, indent=2)  # Convert to formatted JSON string

pdf = PDFDocument(path = "./data/abcb-housing-provisions-2022-20230501b.pdf")
#%%

# Your table data
table_data = [
    ['Element', 'Symbol used in Figure\n5.4.2c', 'Thickness of wall (T)', None, None, None],
    [None, None, '90', '110', '140', '190'],
    ['Return length (minimum)', 'R', '450', '450', '–', '–'],
    ['Spacing of returns\n(maximum) (N2)', 'S', '1050', '1300', '–', '–'],
    ['Spacing of returns\n(maximum) (N3)', 'S', '600', '750', '–', '–'],
    ['Height (maximum)', 'H', '2400', '2400', '1700 (N2)', '2300 (N2)'],
    ['Table Notes\n(1) Dimensions are in mm.\n(2) Return supports are not required for 140 mm and 190 mm thick walls.', None, None, None, None, None]
]

#%%
# Process the table data
table_json = process_table_to_json(tables[2])

# Output the JSON
print(table_json)

# %%

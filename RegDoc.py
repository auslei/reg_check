# %%
import os
import requests
import pdfplumber
import json
import logging
import re
import pandas as pd

from io import BytesIO
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
    def __init__(self, bbox=(0, 60, 580, 820), path=None):
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
        self.pages = {"number": [], "content": [], "n_tables": [], 
                      "tables": [], "references": [], "bullets" : [],
                      "figures": [], "raw_text": [], "part": []}

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
        part = ""
        prev_part = ""
        part_pattern = r'^Part \d+\.\d+ [A-Za-z ]+(?!.*\.{3,})'
        for page in tqdm(self.pdf.pages):
            page_number += 1
            self.pages["number"].append(page_number)
            tables = self._get_page_table(page) 
            n_tables = len(tables)
            content = self._get_page_text(page)
            raw_text = self._get_page_raw_text(page)
            self.pages['number']
            self.pages['n_tables'] = n_tables
            self.pages["tables"].append(tables)
            self.pages["content"].append(content)
            self.pages["references"].append(self._get_page_references(content))
            self.pages["figures"].append(self._get_figures_with_descriptions(content))
            self.pages["raw_text"].append(raw_text)
            self.pages["bullets"].append(self._get_page_bullet_points(raw_text))

            # Find part heading in the text
            heading_match = re.search(part_pattern, raw_text)
            if heading_match:
                part = heading_match.group()
                if prev_part!=part:
                    prev_part = part
            else:
                if prev_part!= "":
                    part = prev_part
                else:
                    part = "N/A"

            self.pages['part'].append(part)
        
        self.df = pd.DataFrame(self.pages)

    def _get_page_text(self, page):
        """Extract text from the pdf page referenced"""
        page_cropped = self._get_cropped_page(page)

        # Find tables and get their bounding boxes
        table_objs = page.find_tables()
        table_areas = [table.bbox for table in table_objs]  # List of table bounding box areas

        extracted_text = []

        # Extract words
        words = page_cropped.extract_words()
        
        # text = page.extract_text()

        def is_inside_table(word_bbox, table_bbox):
            # Check if word bounding box is inside table bounding box
            return (table_bbox[0] <= word_bbox[0] <= table_bbox[2] and
                    table_bbox[1] <= word_bbox[1] <= table_bbox[3])

        # Filter out words that are inside any table area
        for word in words:
            word_bbox = (word['x0'], word['top'], word['x1'], word['bottom'])
            if not any(is_inside_table(word_bbox, table_area) for table_area in table_areas):
                extracted_text.append(word['text'])

        return ' '.join(extracted_text)

    def _get_page_raw_text(self, page):
        """Get raw text for reference"""
        page_cropped = self._get_cropped_page(page)
        return page_cropped.extract_text()

    def _get_cropped_page(self, page):
        """Crop page based on self.bbox. Handles when the page may be landscape"""
        left, top, right, bottom = self.bbox
        p_left, p_top, p_right, p_bottom = page.bbox
 
        if p_right > right and p_bottom > bottom:  #if the bbox fit in the page
            page_cropped = page.crop(self.bbox)
        elif p_bottom > p_right and p_right > bottom: # if the bbox firt in landscape of the page
            page_cropped = page.crop((left, top, bottom, right))
        else:
            page_cropped = page

        return page_cropped

    def _get_page_bullet_points(self, text):
        """use regexp to retrieve all bullet points on the page"""

        # Regular expression pattern for bullet points including multiline
        bullet_pattern = r'^\s*(?:\(\w+\)|\(\d+\)|\d+\)|\d+\.)\s.*?(?=\n\s*(?:\(\w+\)|\(\d+\)|\d+\)|\d+\.|$))'

        # Find all matches in the text
        bullet_points = re.findall(bullet_pattern, text, re.DOTALL | re.MULTILINE)

        # Clean and format the extracted bullet points
        cleaned_bullet_points = [' '.join(point.split('\n')) for point in bullet_points]

        return cleaned_bullet_points

    def _get_page_table(self, page):
        """get list all tables (text) from the page"""
        return page.extract_tables()

    def _get_page_references(self, text):
        # Regular expressions for various reference patterns
        part_ref_pattern = r'Part \d+\.\d+'  # Matches references like "Part 12.2"
        as_nzs_ref_pattern = r'AS/NZS \d+\.\d+'  # Matches references like "AS/NZS 1170.1"
        table_ref_pattern = r'Table \d+\.\d+\w*'  # Matches references like "Table 11.2.2a"
        doc_ref_pattern = r'ABCB Housing Provisions Standard \d+ \(\d+ [a-zA-Z]+ \d+\)'  # Matches specific document references

        # Find all matches in the text
        part_refs = re.findall(part_ref_pattern, text)
        as_nzs_refs = re.findall(as_nzs_ref_pattern, text)
        table_refs = re.findall(table_ref_pattern, text)
        doc_refs = re.findall(doc_ref_pattern, text)

        # Combine all references into a single list
        all_refs = part_refs + as_nzs_refs + table_refs + doc_refs

        # Remove duplicates and return
        return list(set(all_refs))

    def _get_figures_with_descriptions(self, text):
        # Regular expression pattern to capture lines with figures and their descriptions
        figure_pattern = r'(Figure \d+\.\d+\.\d+: [^\n]+)'

        # Find all matches in the text
        figures_matches = re.findall(figure_pattern, text)

        # Dictionary to store figures and their descriptions
        figures_with_descriptions = {}

        for match in figures_matches:
            # Split the match into figure number and description
            figure, description = match.split(': ', 1)
            figures_with_descriptions[figure] = description

        return figures_with_descriptions


    def extract_requirements(self, part_name = "Part 11.2 Stairway and ramp construction"):
        text = "\n".join(self.get_part_content(part_name=part_name))

        # extract requirements from each page
        heading_pattern = r'^(\d+\.\d+\.\d+ [A-Za-z ]+)$'

        # Initialize JSON structure
        data_json = {}

        # Find all headings in the text
        headings = re.findall(heading_pattern, text, re.MULTILINE)

        for heading in headings:
            # Define the section for this heading
            start_idx = text.find(heading) + len(heading)
            end_idx = len(text)
            for subsequent_heading in headings:
                if subsequent_heading != heading:
                    subsequent_idx = text.find(subsequent_heading, start_idx)
                    if subsequent_idx != -1:
                        end_idx = min(end_idx, subsequent_idx)
                        break

            # Extract text under this heading
            section_text = text[start_idx:end_idx]

            # Add to JSON structure under the heading
            data_json[heading] = section_text

        return data_json


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

    def get_part_list(self):
        return self.df.part.unique()
    
    def get_part_content(self, part_name = "Part 11.2 Stairway and ramp construction"):
        return self.df[self.df.part == part_name]["raw_text"]


pdf = PDFDocument(path = "./data/abcb-housing-provisions-2022-20230501b.pdf")
#%%
def extract_requirements(text):
    """
    Extracts top-level bullet points from the provided text, including any sub-level bullets.
    The first occurring bullet format (number, letter, or Roman numeral) is treated as the top level.
    """
    # Define a pattern to match the first occurring bullet format and any subsequent content
    pattern = r'\(([a-zA-Z0-9]+)\)\s(.*?)(?=\([a-zA-Z0-9]+\)\s|\Z)'

    # Use regex to find all matches
    matches = re.findall(pattern, text, re.DOTALL)

    # Process each match to include sub-level bullet points
    bullet_points = []
    for match in matches:
        # Replace line breaks for better readability and concatenate the bullet point with its content
        bullet_point = "(" + match[0] + ") " + match[1].replace('\n', ' ').strip()
        # Add the processed bullet point to the list
        bullet_points.append(bullet_point)

    return bullet_points

#%%
import ollama

def summarise_requirements(text):
    messages = [{'role': 'system',
                 'content': 
                    'You are an experienced requirement provider. You will receive a list of requirement,'
                    ' and you will provide a summary of requirements in bulletpoint form. Please make sure'
                    ' use concise language and all specific requirements are captured'}]
    
    messages.append({'role': 'user', 'content': 'Please provide a summary of the following requirements:\n' + text})

    response = ollama.chat(model='mistral', messages=messages, stream=True, options={
            'temperature': 0.2
        })
    return response

text = df[df.number == 110].raw_text.values[0]

response = summarise_requirements(text)

for c in response:
    print(c['message']['content'], end='', flush=True)
#%%
def extract_table(text):
    messages = [{'role': 'system',
                 'content': 
                'You are an assistant helping extract tables from provided text, please extract table reference'
                ' and table data from the given text in json form.'}]
    messages.append({'role': 'user', 'content': 'Please extract table data from following text: ' + text})

    response = ollama.chat(model='mistral', messages=messages, stream=True)
    return response

text = df[df.number == 110].raw_text.values[0]
text = pdf.pdf.pages[109].extract_text(layout=True)

response = extract_table(text)
full_response = ""

for c in response:
    print(c['message']['content'], end='', flush=True)
    full_response += c['message']['content']
# %%
def table_to_requirements(text):
    messages = [
        {
            'role': 'system',
            'content': (
                'You are an assistant helping convert tables to bullet points of requirements, where '
                'the tables are typically used to look up values. You will be provided a table in markdown '
                'format and you will provide a bullet point of requirements, focusing on look up condition '
                'and values. If there are table notes, include them in a separate bullet point. '
            )
        }
    ]

    messages.append({'role': 'user', 'content': 'Please convert the following table: ' + text})

    response = ollama.chat(model='llama3.1', messages=messages, stream=True)
    return response

response = table_to_requirements(full_response)
for c in response:
    print(c['message']['content'], end='', flush=True)
    full_response += c['message']['content']

# %%

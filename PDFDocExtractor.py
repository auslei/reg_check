# %%
import os
import requests
import pdfplumber
import logging
import re
import pandas as pd

from io import BytesIO
from tqdm import tqdm

# Set up basic configuration for logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class PDFDocument:
    def __init__(self, path, bbox=(0, 60, 580, 820)):
        if bbox:
            self.bbox = bbox

        if os.path.isfile(path):
            self._load_pdf_from_file(path)
        elif path.startswith(('http://', 'https://')):
            self._load_pdf_from_url(path)
        else:
            logging.error(f"Invalid path or URL: {path}")
            return
        
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
        self.df = pd.DataFrame(columns = ['part', 'page_number', 'text'])
        page_number = 0
        part = ""
        prev_part = ""
        part_pattern = r'^Part \d+\.\d+ [A-Za-z ]+(?!.*\.{3,})'
        part_id_pattern = r'(\d+\.\d+)'

        for page in self.pdf.pages:
            page_number += 1
            raw_text = self._get_page_text(page)
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

            if part!="N/A":
                new_row = pd.DataFrame({'part': [part], 'page_number': [page_number], 'text': [raw_text]})
                self.df = pd.concat([self.df, new_row], ignore_index=True)

            if page_number == 412: break # definition section
        self.df['wc'] = self.df['text'].map(lambda x: len(str(x).split())) # generate a wordcount
        self.df['part_id'] = self.df['part'].map(lambda x: re.search(part_id_pattern, x).group(1))

        self._process_requirements()


    def _process_requirements(self):
        self.df_parts = self.df.groupby("part_id")['text'].apply("\n".join).reset_index()

        def extract_requirements(text):
            # extract requirements from each page
            reqirement_pattern =  r'^(\d+\.\d+\.\d+ [A-Za-z -]+)$'  #r'^(\d+\.\d+\.\d+ [A-Za-z ]+)$'

            # Find all headings in the text
            requirements = re.findall(reqirement_pattern, text, re.MULTILINE)

            reqs = []
            req_texts = []

            for requirement in requirements:
                # Define the section for this heading
                start_idx = text.find(requirement) + len(requirement)
                end_idx = len(text)
                for subsequent_requirement in requirements:
                    if subsequent_requirement != requirement:
                        subsequent_idx = text.find(subsequent_requirement, start_idx)
                        if subsequent_idx != -1:
                            end_idx = min(end_idx, subsequent_idx)
                            break

                # Extract text under this heading
                requirement_text = text[start_idx:end_idx]

                reqs.append(requirement)
                requirement_text = text[start_idx:end_idx]
                req_texts.append(requirement_text)

            return reqs, req_texts
        
        req_id_pattern = r'(\d+\.\d+\.\d+)'
        self.df_parts['requirement'], self.df_parts['requirement_text'] = zip(*self.df_parts['text'].apply(extract_requirements))
        self.df_requirements = self.df_parts.explode(['requirement', 'requirement_text'])[["part_id", "requirement", "requirement_text"]]
        self.df_requirements["wc"] = self.df_requirements['requirement_text'].map(lambda x: len(str(x).split()))
        #self.df_requirements['req_id'] = self.df_requirements['requirement'].map(lambda x: re.search(req_id_pattern, x).group(1))
        self.df_parts = self.df_parts[["part_id", 'text']]
        self.df_parts["wc"] = self.df_parts.text.map(lambda x: len(str(x).split()))

    def _get_page_text(self, page):
        if self.bbox:
            return self._get_cropped_page(page).extract_text()
        else:
            return page.extract_text()

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


pdf = PDFDocument(path = "./data/abcb-housing-provisions-2022-20230501b.pdf")

#%%
import pickle

with open('./data/processed_doc.pickle', 'wb') as f:
    pickle.dump(obj = pdf.df_requirements, file = f)




# %%
prompt_template = PromptTemplate(template="""[INST]
The following is a requirement containing a set of points to meet. Each point may reference a figure, a table or another section.  It may also contain a reference to section ID published at the beginning of the test in the format of  [Year: Section ID] 
Please extract the text enclosed between ~  below into the below, where possible, consolidate the bullet points:
- Object
- Location
- Relationship to another object
- Requirement
- References
- Old Section

~
[2019: 3.9.1.3]
An external ramp serving an external doorway or a ramp within a building must—
(a) be designed to take loading forces in accordance with AS/NZS 1170.1; and
(b) have a gradient not steeper than 1:8; and
(c) be provided with landings complying with 11.2.5 at the top and bottom of the ramp and at intervals not greater
than 15 m.
Notes: Livable housing design
Where an external ramp is provided for the purposes of compliance with the ABCB Standard for Livable Housing Design,
the requirements of that Standard apply.
Explanatory Information
In relation to external ramps, 11.2.3 applies to a ramp serving an external door. For the purpose of 11.2.3 a driveway is
not considered to be a ramp.
~

Result:
Object: Ramp
Location: External Doorway or Within a Building
Relationship to another object: N/A
Requirement: take loading forces in accordance with AS/NZS 1170.1; have a gradient not steeper than 1:8; landings comply with 11.2.5 at top of the ramp and at intervals not greater than 15 m.
References: AS/NZS 1170.1, 11.2.3
Old Section: 2019.3.9.13

The following is a requirement containing a set of points to meet. Each point may reference a figure, a table or another section.  It may also contain a reference to section ID published at the beginning of the test in the format of  [Year: Section ID] 

Please extract the text enclosed between ~  below into the below, where possible, consolidate the bulet points:
- Object
- Location
- Relationship to another object
- Requirement
- References
- Old Section

~
{text}
~
Result:
[/INST]""" 
                                 , input_variables=["text"])



sample = """[2019: 3.9.1.3]
An external ramp serving an external doorway or a ramp within a building must—
(a) be designed to take loading forces in accordance with AS/NZS 1170.1; and
(b) have a gradient not steeper than 1:8; and
(c) be provided with landings complying with 11.2.5 at the top and bottom of the ramp and at intervals not greater
than 15 m.
Notes: Livable housing design
Where an external ramp is provided for the purposes of compliance with the ABCB Standard for Livable Housing Design,
the requirements of that Standard apply.
Explanatory Information
In relation to external ramps, 11.2.3 applies to a ramp serving an external door. For the purpose of 11.2.3 a driveway is
not considered to be a ramp."""

llmchain = LLMChain(llm = llm, prompt = prompt_template, verbose = True)

print(llmchain.run(text = sample))


# %%
print(prompt_template.format(text = sample))
# %%
req_summary_template = ("[INST]The following text contains requirements for building compliance. "
                        "Please summarize the requirement to be within 200 tokens and highlight what this requirement is about.\n"
                        "Requirement:\n"
                        "{req}\N"[/INST]")

req_prompt_template = PromptTemplate(template=req_summary_template, input_variables=["req"])

llm_chain = LLMChain(llm = llm, prompt = req_prompt_template, verbose=True)
print(llm_chain.run(req = sample))
# %


# %%
from llama_cpp import Llama
from langchain.llms import LlamaCpp, OpenAI, HuggingFaceHub
from langchain import PromptTemplate
from langchain.chains import LLMChain

def get_llm(option = "llama"):
    model_path = "./models/ggml-llama-7b-chat-q4_0.bin"
    OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
    if option == "llama":
        llm = LlamaCpp(model_path="./models/ggml-llama-7b-chat-q4_0.bin", 
                    n_ctx = 1024,
                    n_gpu_layers = 1,
                    temperature = 0.2,
                    use_mlock = True,
                    top_k = 10,
                    top_p = 0.8,
                    max_tokens = 512,
                    repeat_penalty = 1.5,
                    verbose = False)
    else:
        # OpenAI llm
        llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_KEY)
    return llm

def summarise(llm = llm, req_text):
    req_summary_template = ("[INST]The following text contains requirements for building compliance. "
                        "Please summarize the requirement to be within 200 tokens and highlight what this requirement is about.\n"
                        "Requirement:\n"
                        "{req}\N[/INST]")

    req_prompt_template = PromptTemplate(template=req_summary_template, input_variables=["req"])

    llm_chain = LLMChain(llm = llm, prompt = req_prompt_template, verbose=True)

    return llm_chain.run(req = req_text)


df =pdf.df_requirements
# %%

#%%

#!/usr/bin/env python # [1]
"""This code does pre-processing of the regulatory documentation for NCC volume 2
"""
import os   # standard library, [3]
import sys

path = "./data/ncc2022-volume-two-20230501b.pdf"

#%%
reader = PdfReader(path)
number_of_pages = len(reader.pages)

pages = []

for page in reader.pages:
    text = page.extract_text()
    pages.append(text)

print(len(pages))

#%%
print(pages[80])

#%%
from tabula import read_pdf
# %%
tables = read_pdf(path)
# %%
from langchain.document_loaders import BSHTMLLoader
loader = BSHTMLLoader("https://python.langchain.com/docs/modules/data_connection/document_loaders/html")
data = loader.load()
data
# %%

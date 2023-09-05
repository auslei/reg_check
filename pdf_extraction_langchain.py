#%%
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PDFMinerLoader
from langchain.document_loaders import pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

import chromadb

pdf_path = "./data/abcb-housing-provisions-2022-20230501b.pdf"

loader = PyPDFLoader(pdf_path)
splitter = RecursiveCharacterTextSplitter(
                        chunk_size = 200,
                        chunk_overlap  = 50,
                        length_function = len,
                        is_separator_regex = False)

pages = loader.load()

# %%
client = chromadb.PersistentClient("./db") # intiate a chroma db

collection_name = "abcb_housing_provisions_ex"
client.delete_collection(name = collection_name)

model_name = "all-MiniLM-L6-v2" 
embedding_fn = SentenceTransformerEmbeddings(model_name = model_name)

vectordb = Chroma.from_documents(documents = pages, embedding = embedding_fn,
                                 collection_name = collection_name,
                                 client = client)
# %%
result = vectordb.similarity_search("what is the regulation related to staircase?")

for r in result:
    print(r.page_content)
# %%
vectordb.delete_collection(collection_name)
# %%

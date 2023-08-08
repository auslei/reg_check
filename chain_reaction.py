#%% load chroma database
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import chromadb
import llm_models 

# retrieve the Chroma db for langchain
def get_vectordb(chroma_path = "./db", 
                 model_name = "all-MiniLM-L6-v2", 
                 collection_name = "abcb_housing_provisions"):
    embeddings = SentenceTransformerEmbeddings(
        model_name = model_name,
    )
    client = chromadb.PersistentClient(path=chroma_path)
    vectordb = Chroma(client = client, collection_name = collection_name, 
                    embedding_function= embeddings)

    return vectordb

# search the database, return a result
# 0 - Similarity Search
# 1 - Max Marginal Relevance Search
# 2 - 
def search_db(db, query, search_type = 0, ):
    if search_type == 0:
        r = db.similarity_search(query)
    elif search_type == 1:
        r = db.max_marginal_relevance_search(query)
    elif search_type == 2:
        model_name = "google/flan-t5-base"
        llm = HuggingFacePipeline(pipeline = \
                                  llm_models.get_llm_pipeline(model_name = model_name))
        compressor = LLMChainExtractor.from_llm(llm = llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=db.as_retriever(search_type = "mmr")
        )
        r = compression_retriever.get_relevant_documents(query)
    return r

# Pretty print for testing purpose
def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".\
          join([f"Document {i+1}) (Clause {d.metadata['clause_number']}):\n\n" \
                + d.page_content[:100] for i, d in enumerate(docs)]))

#%%
# test code
db = get_vectordb()
r = search_db(db = db, query = "What is the requirement of site earthworks.", search_type = 2)
pretty_print_docs(r)
# %%

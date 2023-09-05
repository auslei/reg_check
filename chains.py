#%% load chroma database
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import chromadb
import models 


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
def search_db(db: Chroma, query, search_type = 0):
    # if search type is 0 then perform simple db query
    if search_type == 0:
        #retriver = db.as_retriever(search_type = "similarity_score_threshold", search_kwargs={"score_threshold": .5, "k": 3})
        retriver = db.as_retriever(search_type = "mmr", search_kwargs = {"k": 3}) # use max margine retriver
    elif search_type == 1:
        llm = models.get_langchain_llama_model(chat_mode = False)
        compressor = LLMChainExtractor.from_llm(llm = llm)
        retriver = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=db.as_retriever(search_type = "mmr")
        )
    r = retriver.get_relevant_documents(query)
    
    return r

# Pretty print for testing purpose
def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".\
          join([f"Document {i+1}) (Clause {d.metadata['clause_number']}):\n\n" \
                + d.page_content[:100] for i, d in enumerate(docs)]))

# get all related documents from the results
def get_related_results(results):
    clause_numbers = set()
    for r in results:
        clause_numbers.add(r.metadata['clause_number'])
    
    return clause_numbers


#%%
# test code
def test():
    db = get_vectordb()
    results = search_db(db = db, query = "What is the requirement of staircase rise and going", search_type = 1)
    #pretty_print_docs(r)

    import models as lm

    model = lm.get_llama_model(context_windows=1200)
    tokenizer = lm.get_llama_tokenizer()

    for result in results:
        print(lm.extract_regulations(text=result.page_content, model=model))
    # %%
    print(lm.extract_regulations(model = model, text = results[0].page_content))
    # %%
    for r in results:
        tokens = tokenizer.encode(r.page_content)
        print(len(tokens))
# %%
prompt_template = """
    [INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

    Please summarize below text in bullet form, showing values in markdown format:

    {text} [/INST]
    """
# %%

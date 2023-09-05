import streamlit as st
import models as m
import chains as c
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# gets the model and db
#llm = m.get_langchain_llama_model(context_windows=4096, chat_mode=True)
#llm = m.get_langchain_t5_model()
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

#What is the core requirement for stairway construction?

#*******************
#* LLama 2 Prompts *
#*******************
sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your should follow instructions and provide concise and accurate information.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """

instruction = """/n/nINSTRUCTION: Summarise essential requirements, including information from tables from below text:/n\n\nTEXT: {text}"""
instruction2 = """/n/nINSTRUCTION: Consolidate text below in bullet form, remove duplicate and use complete formal English, output using markdown format:/n\n\nTEXT: {text}"""

db = c.get_vectordb()

template = \
"""
Summarise essential requirements in bullet point form from below text, also identify any tables and extract the tables in markdown form:

{text}"""

template2 = \
"""
Consolidate and summarise the information below in bullet form, remove duplicate and use the least words, output using markdown:

{text}
"""

template = m.get_llama_prompt(instruction, sys_prompt)
template2 = m.get_llama_prompt(instruction2, sys_prompt)

prompt_template = PromptTemplate(input_variables=['text'], template = template)
prompt_template2 = PromptTemplate(input_variables=['text'], template = template2)

# contruction a LLM chain

#qa_chain = RetrievalQA()

st.title("🤖Building Code🏘️")
prompt = st.text_input(label = "Enter your query:") #value = "What is the requirements for stairways?"

# chain_type is one of "stuff", "map_reduce", "map-rerank"
#qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=db.as_retriever(), verbose = True)

combined_regulations = []

if prompt:
    response = c.search_db(db, query=prompt, search_type=0)
    for d in response:
        with st.spinner("Please wait, I am thinking..."):
            llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True) 
            regulations = llm_chain.run(text = d.page_content)
            combined_regulations.append(regulations)
            with st.expander("# " + d.metadata["clause_number"] + " " + d.metadata["clause_title"] + "(expand to see full details)"):
                st.markdown(regulations)
    
    llm_chain = LLMChain(llm=llm, prompt=prompt_template2, verbose=True)
    st.subheader("Summary")
    with st.spinner("Generating Summary, please wait..."):
        st.markdown(llm_chain.run("\n".join(combined_regulations)))

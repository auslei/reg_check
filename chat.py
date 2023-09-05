import streamlit as st
import chains as cr
import models as lm

DB = cr.get_vectordb()
model = lm.get_llama_model(context_windows = 2048)

st.title = "Regulatory Compliance Bot"
st.header = "Regulatory Compliance Bot"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter a query..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        st.write("Please wait, I am preparing results...")

    r = cr.search_db(db = DB, query = prompt, search_type = 2)
    print(r)
    response = f"{r}"
    requirements = []
    # Display assistant response in chat message container
    st.write()

    #if inference_type == "Requirements":
    with st.chat_message("assistant"):
        st.write(f"I have found {len(r)} clauses, and they are listed below.")
        for d in r:         
            with st.spinner("Please wait, I am thinking..."):   
                with st.expander("# " + d.metadata["clause_number"] + " " + d.metadata["clause_title"] + "(expand to see full details)"):
                    st.write(d.page_content)
                st.markdown(lm.extract_regulations(model = model, text = d.page_content))

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


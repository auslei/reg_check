from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import LlamaCpp, OpenAI, HuggingFaceHub
from llama_cpp import Llama
from langchain import PromptTemplate
import os
import torch

# API keys
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
HF_HUB_TOKEN = os.environ.get("HUGGINGFACE_ASSETS_HUB_TOKEN")
GOOGLE_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")


#%% Summarisation models
def split_tensor_with_overlap(tensor, row_size=1024, overlap=0):
    """
    Split a 1D torch tensor into multiple rows, each of size row_size.
    Rows will overlap by the specified amount.
    If the final row is not of size row_size, it will be padded with 0s.
    
    Parameters:
    - tensor: The 1D torch tensor to be split
    - row_size: The desired size of each row
    - overlap: Number of elements by which rows should overlap
    
    Returns:
    - A 2D tensor with rows of size row_size
    """
    step = row_size - overlap
    num_rows = 1 + (max(0, len(tensor) - row_size) + step - 1) // step
    result = torch.zeros(num_rows, row_size, dtype=tensor.dtype)
    
    #print(step, num_rows, tensor.shape)

    for i in range(num_rows):
        start_idx = i * step
        end_idx = start_idx + row_size
        result[i, :min(row_size, len(tensor) - start_idx)] = tensor[start_idx:end_idx]
    
    return result

#%%
# This summarises a text using the facebook bart large cnn, 
# for large document it will exceed the max context window size for the model,
# to handle this, we split the content to multiple windows and generate
def summarise(text, contenx_window = 512, overlap = 100):
    model_name = "facebook/bart-large-cnn"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokeniser = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    input_ids = tokeniser.encode(text, return_tensors = "pt").squeeze()

    input_ids_chuncks = split_tensor_with_overlap(input_ids, row_size = contenx_window, overlap = overlap) # this is to split the large document into chuncks
    print(input_ids_chuncks.shape)

    summaries = []
    for input in input_ids_chuncks:
        output = tokeniser.decode(
                    model.generate(input.unsqueeze(0), length_penalty=3.0,
                                      min_length=30,
                                      max_length=100)[0]
                    , skip_special_tokens= True)
        
        summaries.append(output)

    return summaries

def get_langchain_t5_model():
    llm = HuggingFaceHub(repo_id='google/flan-t5-base', model_kwargs={'temperature':1e-10})
    return llm



#**************************************
#*   LLama 2                          *
#**************************************

## Default LLaMA-2 prompt style
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def get_llama_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def get_llama_model_path(chat_mode = True):
    if chat_mode == False:
        model_path = "./models/ggml-llama-7b-q4_0.bin"
    else:
        model_path = "./models/ggml-llama-7b-chat-q4_0.bin"
    return model_path

# This gets LlamaCpp object from Langchain
def get_langchain_llama_model(context_windows = 2048, 
                              use_gpu = True,
                              use_mlock = True, 
                              repeat_penalty = 1.5, 
                              temperature = 0,
                              chat_mode = True) -> LlamaCpp:
    model_path = get_llama_model_path(chat_mode = chat_mode) 
    llm = LlamaCpp(model_path = model_path, 
                   n_ctx = context_windows,
                   n_gpu_layers = use_gpu, 
                   use_mlock = use_mlock, 
                   temperature=temperature,
                   repeat_penalty = repeat_penalty)
    return llm

# This gets a llama model, by default it will use the chat model.
def get_llama_model(context_windows = 2048, 
                    use_gpu = True, 
                    use_mlock = True, 
                    chat_model = True) -> Llama:
    if chat_model == False:
        model_path = "./models/ggml-llama-7b-q4_0.bin"
    else:
        model_path = "./models/ggml-llama-7b-chat-q4_0.bin"

    return Llama(model_path = model_path, 
                 n_ctx = context_windows, 
                 n_gpu_layers = use_gpu, 
                 use_mlock = use_mlock)


## TODO: refactor
def extract_regulations(text: str, 
                        model, 
                        max_tokens = 256) -> str:
    prompt_template = """
    [INST]
    Please summarize below text in markdown bullet form, showing only requirements:

    {text} [/INST]
    """

    prompt = prompt_template.format(text = text)

    output = model(prompt = prompt, max_tokens = max_tokens, temperature = 0.2, repeat_penalty = 1.5)
    print(output)

    return(output)


## TODO: refactor
def validate_statement(context, statement, model, max_tokens = 200):
    #You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    prompt_template = """
    [INST] <<SYS>>
    Given the below context:
    {context}

    Please validate below statement and provide reasoning:

    {statement} [/INST]
    """

    prompt = prompt_template.format(context = context, statement = statement)

    output = model(prompt = prompt, max_tokens = max_tokens, temperature = 0.2)

    return(output["choices"][0]['text'])

# not sure how to use the llama cpp to obtain tokenizer, using huggingface instead.
# unsure why that api token only works for use_auth_token argument. 
def get_llama_tokenizer():
    import os
    from transformers import AutoTokenizer

    # Set environment variables
    hf_auth = os.getenv("HUGGINGFACE_ASSETS_HUB_TOKEN")
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token = hf_auth)
    return tokenizer


def get_flant5_model(max_length = 100, temperature = 0.2):
    model_name = "google/flan-t5-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name) #load model and send to MBP GPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #template = """Extract requirements from the following text: {text}"""

    llm = pipeline("text2text-generation",
                model=model, 
                tokenizer=tokenizer, 
                max_length=max_length,
                temperature = temperature)
    return llm


# %%

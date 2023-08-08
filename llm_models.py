#%%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
import torch

# take a model_name and produce a transformer pipeline
# return None if model is not found
def get_llm_pipeline(model_name):
    pipe = None
    if model_name == "google/flan-t5-base":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        pipe = pipeline(
            "text2text-generation",
            model=model, 
            tokenizer=tokenizer, 
            max_length=100
        )
    return pipe

#%%
hf_auth = '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'

model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "tiiuae/falcon-40b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          use_auth_token = hf_auth)

#%%
#model = AutoModelForCausalLM.from_pretrained(model_name)#,
                                             #device_map='auto',
                                             #torch_dtype=torch.float16,
                                             #use_auth_token=hf_auth,
                                             #load_in_8bit=True)#,
                                             #load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b-instruct")
inputs = tokenizer("What's the best way to divide a pizza between three people?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
## %%
from transformers import pipeline

pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )
# %%
pipe("what is up?")
# %%

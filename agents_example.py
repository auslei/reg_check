import ollama


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

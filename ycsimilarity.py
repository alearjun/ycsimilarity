import streamlit as st
import openai
from pinecone import Pinecone
import time
import os

pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# Pinecone setup

environment = 'gcp-starter'
pc = Pinecone(api_key=pinecone_api_key)
index_name = 'yc-companies'
index = pc.Index(index_name)

# OpenAI setup
embed_model = "text-embedding-ada-002"

# Function to retrieve context
def retrieve(query, openai_api_key):
    
    openai.api_key = openai_api_key

    res = openai.embeddings.create(
        input=[query],
        model=embed_model
    )


    xq = res.data[0].embedding
    contexts = []
    time_waited = 0
    while len(contexts) < 3 and time_waited < 60 * 12:
        res = index.query(vector=xq, top_k=5, include_metadata=True)
        contexts += [
            x['metadata']['text'] for x in res['matches']
        ]
        time.sleep(1)
        time_waited += 1

    if time_waited >= 60 * 12:
        return "No contexts retrieved. Try to answer the question yourself!"

    prompt_start = "Answer the question based on the context below. \n\nContext:\n"
    prompt_end = f"\n\nQuestion: What are YC companies that are involved in {query}? \nAnswer:"

    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= 6000:
            prompt = prompt_start + "\n\n---\n\n".join(contexts[:i-1]) + prompt_end
            break
        elif i == len(contexts)-1:
            prompt = prompt_start + "\n\n---\n\n".join(contexts) + prompt_end
    return prompt

# Function to generate completion
def complete(prompt, openai_api_key):

    openai.api_key = openai_api_key

    res = openai.chat.completions.create(
        model='gpt-3.5-turbo',
        temperature=0,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        messages=[
            {"role": "system", 
             "content": """You are a helpful assistant that describes companies that may be similar to a concept that the user enterred. 
             You should respond with a list of companies, their batch, a short description, and their websites. You must always include these four details 
             for each company. lways follow the following format and do not add any text after the list: 
             
             The following YC companies may be similar: 
             * {Company Name} (Batch: {batch})
             {Description}
             * {insert the list of companies here}
             """},
            {"role": "user", "content": prompt}
        ]
    )
    return res.choices[0].message.content

# Streamlit UI
st.title('YC Company Lookup')

st.write("Enter your OpenAI API Key:")
user_openai_api_key = st.text_input("OpenAI API Key", type="password")

user_query = st.text_input("Enter company details:")

if st.button('Submit'):
    if user_query and user_openai_api_key:
        prompt = retrieve(user_query, user_openai_api_key)
        response = complete(prompt, user_openai_api_key)
        st.text("Response:")
        st.write(response)
    else:
        st.write("Please enter both an OpenAI API Key and company details.")

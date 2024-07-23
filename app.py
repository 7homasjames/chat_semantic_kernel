import os
import textwrap
import streamlit as st
from dotenv import load_dotenv
import PyPDF2
import asyncio
import requests
from tqdm import tqdm 
from semantic_kernel.functions import KernelArguments
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings, OpenAITextEmbedding
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin

from prep import get_chunks, get_embeddings, get_pdf_data, embed
from azs import CustomSearchAI

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

def get_url(end_point):
    url = f"http://127.0.0.1:8000/{end_point}/"
    return url


def get_context(user_input):
    # emb = embed(user_input)
    payload = {"query" : user_input}
    print(payload)
    resp = requests.post(get_url("context"), json=payload)
    print(resp.json())
    resp, files,page_no = "\n".join(resp.json()['docs']), resp.json()['filenames'][0], resp.json()['page_number']
    print(page_no)
    

    return resp, files, page_no


def get_response(user_input):
    context, files,page_no = get_context(user_input)
    print(context)
    payload = {
        'query' : user_input,
        'context' : context
    }
    resp = requests.post(get_url("response"), json=payload)
    resp = resp.json()['output']
    return resp, files, page_no



if "files" not in st.session_state.keys():
    st.session_state['files'] = [] 

# if "csai" not in st.session_state.keys():
#     with st.spinner('Connecting to Azure search AI'):
#         st.session_state['csai'] = CustomSearchAI()
url = "http://127.0.0.1:8000/push_docs/"

with st.sidebar:
    st.title('Semantic Kernel Question Answering System 💬')
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if 'files' not in st.session_state:
            st.session_state['files'] = []

        with st.spinner('Indexing the documents'):
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state['files']:
                    text_data = get_pdf_data(uploaded_file)
                    chunks = get_chunks(text_data)
                    docs = [get_embeddings(txt, uploaded_file.name, page_num) for txt, page_num in chunks]
                    print(docs)
                    for i in tqdm(range(0, len(docs), 10)):
                        payload = {"items": docs[i:i+10]}
                        response = requests.post(get_url("push_docs"), json=payload)

                    st.session_state['files'].append(uploaded_file.name)


            



# Function to display user messages with rounded rectangle borders
def user_message(message):
    with st.chat_message("user"):
        st.write(message)
    # st.markdown(f'<div class="user-message" style="display: flex; justify-content: flex-end; padding: 5px;">'
    #             f'<div style="background-color: #196b1c; color: white; padding: 10px; border-radius: 10px; font-size:18px; margin-bottom:10px; margin-left:20px;">{message}</div>'
    #             f'</div>', unsafe_allow_html=True)

# Function to display bot messages with rounded rectangle borders
def bot_message(message):
    with st.chat_message("assistant"):
        st.write(message)

    # st.markdown(f'<div class="bot-message" style="display: flex; padding: 5px;">'
    #             f'<div style="background-color: #074c85; color: white; padding: 10px; border-radius: 10px; font-size:18px; margin-bottom:10px; margin-right:20px;">{message}</div>'
    #             f'</div>', unsafe_allow_html=True)


# Initialize chat history using session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input field for user to enter a message
user_input = st.chat_input("Your Message:")
# Button to send the user's message
if user_input:
    # Display previous chat messages
    for message, is_bot_response in st.session_state.chat_history:
        if is_bot_response:
            bot_message(message)
        else:
            user_message(message)
    # Add the user's message to the chat history
    st.session_state.chat_history.append((user_input, False))

    # Display the user's message
    user_message(user_input)

    # Bot's static response (you can replace this with a dynamic response generator)
    resp, files, page_number =  get_response(f"{user_input}")#"This is a static bot response."
    bot_response = resp + f"\n\nReferences :\n\n {files}" + f"\n\nPage_number :\n\n {page_number}"
    # Add the bot's response to the chat history
    st.session_state.chat_history.append((bot_response, True))
    
    # Display the bot's response
    bot_message(bot_response)
    

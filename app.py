# Import streamlit for app dev
import streamlit as st

# Import transformer classes for generation
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
# Import torch for datatype attributes 
import torch
# Import the prompt wrapper...but for llama index
from llama_index.prompts.prompts import SimpleInputPrompt
# Import the llama index HF Wrapper
from llama_index.llms import HuggingFaceLLM
# Bring in embeddings wrapper
from llama_index.embeddings import LangchainEmbedding
# Bring in HF embeddings - need these to represent document chunks
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# Bring in stuff to change service context
from llama_index import set_global_service_context
from llama_index import ServiceContext
# Import deps to load documents 
from llama_index import VectorStoreIndex, download_loader
from pathlib import Path
import tempfile

# Define variable to hold llama2 weights naming 
name = "model_name from huggingface or your own trained model_path"
# Set auth token variable from hugging face 
auth_token = "your huggingface auth_token(write)"

@st.cache_resource
def get_tokenizer_model():
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', use_auth_token=auth_token)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir='./model/'
                            , use_auth_token=auth_token, torch_dtype=torch.float16, 
                            rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=True, device_map='auto') 

    return tokenizer, model
tokenizer, model = get_tokenizer_model()

# Create a system prompt 
system_prompt = """<s>[INST] <<SYS>>
당신은 도움이 되고, 예의바르고, 정직한 보조자입니다. 항상 가능한 한 도움이 되도록 대답해 주세요, 단 안전하게. 당신의 답변에는 해로운, 비윤리적인, 인종차별적인, 성차별적인, 독성 있는, 위험한 또는 불법적인 내용이 포함되어서는 안됩니다.
응답이 사회적으로 편견이 없고 긍정적인 성격을 가지도록 해 주세요.

질문이 아무런 의미가 없거나 사실적으로 일관성이 없다면, 왜 그런지를 설명해 주세요, 대신 틀린 것을 대답하지 마세요. 질문에 대한 답을 모른다면, 거짓 정보를 공유하지 마세요.

당신의 목표는 PDF 형식의 문서와 관련된 모든 것에 대한 답을 제공하는 것입니다.<</SYS>>
"""
# Throw together the query wrapper
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

# Create a HF LLM using the llama index wrapper 
llm = HuggingFaceLLM(context_window=2048,
                    max_new_tokens=1024,
                    system_prompt=system_prompt,
                    query_wrapper_prompt=query_wrapper_prompt,
                    model=model,
                    tokenizer=tokenizer)

# Create and dl embeddings instance  
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Create new service context instance
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    chunk_overlap=512,
    llm=llm,
    embed_model=embeddings
)
# And set the service context
set_global_service_context(service_context)

# Download PDF Loader 
PyMuPDFReader = download_loader("PyMuPDFReader")
# Create PDF Loader
loader = PyMuPDFReader()

# File uploader for the user to upload PDF file
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

if uploaded_file is not None:
    # Create a temporary file to store the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Load documents 
    documents = loader.load(file_path=temp_path, metadata=True)

    # Create an index - we'll be able to query this in a sec
    index = VectorStoreIndex.from_documents(documents)
    # Setup index query engine using LLM 
    query_engine = index.as_query_engine()

    # Create centered main title 
    st.title('🦙 Llama RAG for PDF')
    # Create a text input box for the user
    prompt = st.text_input('Input your prompt here')

    # If the user hits enter
    if prompt:
        response = query_engine.query(prompt)
        # ...and write it out to the screen
        st.write(response)

        # Display raw response object
        with st.expander('Response Object'):
            st.write(response)
        # Display source text
        with st.expander('Source Text'):
            st.write(response.get_formatted_sources())

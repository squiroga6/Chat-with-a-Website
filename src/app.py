import streamlit as st

# langchain utils
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# langchain's community packages
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader

# openai specific
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# load env vars
from dotenv import load_dotenv
load_dotenv()

# pipeline
def get_vector_store_from_url(url):
    """
    Get text in document form
    """

    # get document
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter()
    
    # split document into chunks
    document_chunks = text_splitter.split_documents(document)
    
    # creatre vector store
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
      
    return vector_store

def get_context_retriever_chain(vector_store):
    """
    Create all relevant information from prompt
    """
    
    # initialize llm
    llm = ChatOpenAI()    

    # create retriever (allows to retrieve relevant text)
    retreiver = vector_store.as_retriever()
    
    # populate prompt with all chat history: takes an array of messages
    prompt = ChatPromptTemplate.from_messages([
        # add chat history if exists
        MessagesPlaceholder(variable_name="chat_history"),
        # pass a human message as a tuple : type, prompt
        ("user","{input}"),
        # add prompt
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    return create_history_aware_retriever(llm,retreiver,prompt)

def get_conversational_rag_chain(retriever_chain):
    """
    Create a documents chain that takes contxt and answer. Blocking it with a retrieval chain
    """
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system","Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}")
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):

    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history":st.session_state.chat_history,
        "input":user_input
    })
    return response['answer']

# app config
st.set_page_config(page_title = "Chat with websites!",page_icon ="🤖")
st.title("Chat with a website")
    
# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input(label = "Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else:

    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content = "Hello, I am a bot, how can I help you?")
            ]
    
    # create conversation chain
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store_from_url(website_url)
        
    # user input
    user_query = st.chat_input("Type your message here...")
    
    if user_query is not None and user_query != "":
        response = get_response(user_input=user_query)                
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
    
    # converstaion
    for message in st.session_state.chat_history:
        if isinstance(message,AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
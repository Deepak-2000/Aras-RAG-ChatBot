import os
import uuid
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Correct import
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import logging
import json
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
PDF_PATH = 'ArasDocs'
PERSIST_DIR = 'Aras_Database'
CHAT_STORE_FILE = "chat_history.json"

st.set_page_config(page_title='Aras Bot', page_icon='🤖', layout='wide')
st.title('Aras ChatBot')

def save_conversations():
    """Save all conversations to disk."""
    try:
        with open(CHAT_STORE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                st.session_state.conversations,
                f,
                indent=2,
                ensure_ascii=False
            )
    except Exception as e:
        logger.error(f"Error saving conversations: {e}")


def load_conversations():
    """Load conversations from disk."""
    
    if os.path.exists(CHAT_STORE_FILE):
        try:
            with open(CHAT_STORE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data:
                return data

        except Exception as e:
            logger.error(f"Error loading conversations: {e}")

    cid = str(uuid.uuid4())

    return {
        cid: {
            "title": "New Chat",
            "messages": []
        }
    }
def rebuild_history_store(conversations):
    """
    Convert saved conversations back into
    LangChain ChatMessageHistory objects.
    """

    history_store = {}

    for cid, chat in conversations.items():

        history = ChatMessageHistory()

        for msg in chat["messages"]:

            if msg["role"] == "user":
                history.add_message(
                    HumanMessage(content=msg["content"])
                )

            elif msg["role"] == "assistant":
                history.add_message(
                    AIMessage(content=msg["content"])
                )

        history_store[cid] = history

    return history_store
# Initialize session state
if "conversations" not in st.session_state:

    st.session_state.conversations = load_conversations()

    st.session_state.current_chat_id = list(
        st.session_state.conversations.keys()
    )[0]


if "history_store" not in st.session_state:

    st.session_state.history_store = rebuild_history_store(
        st.session_state.conversations
    )

# Initialize embeddings
embeddings = OllamaEmbeddings(model='nomic-embed-text')

# Load or create vector store
@st.cache_resource
def get_vectorstore():
    """Initialize or load the vector store."""
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        logger.info("Loading existing vector store...")
        return Chroma(
            persist_directory=PERSIST_DIR, 
            embedding_function=embeddings
        )
    else:
        logger.info("Creating new vector store from documents...")
        loader = DirectoryLoader(
            PDF_PATH, 
            glob='**/*.pdf', 
            loader_cls=PyMuPDFLoader,
            show_progress=True
        )
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks")
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        
        # Explicitly persist (important for some Chroma versions)
        vectorstore.persist()
        logger.info("Vector store created and persisted")
        
        return vectorstore

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create chat history for a session."""
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = ChatMessageHistory()
    return st.session_state.history_store[session_id]

# Create contextualization chain
contextualize_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Rewrite follow-up questions as standalone questions. Do not answer.'),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
])

history_aware_retriever = create_history_aware_retriever(
    llm, 
    retriever, 
    contextualize_prompt
)

# Create QA chain
qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert Aras documentation assistant.

Instructions:
- Answer only from the provided context.
- Use chat history only to understand references such as "it", "that", or follow-up questions.
- Provide complete and detailed explanations when information is available.
- Explain concepts clearly and in a structured manner.
- For procedures, provide step-by-step instructions.
- For technical topics, include relevant details, prerequisites, limitations, and best practices when present in the context.
- Use bullet points or numbered lists when appropriate.
- Do not invent information or make assumptions.
- If the answer cannot be found in the context, respond:
  "I could not find this information in the provided documentation."
- Be accurate, professional, and helpful.
- Provide Answer from the context or history if not fount say I do not know.
Context:
{context}
"""
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

chat_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)

# Sidebar for conversation management
with st.sidebar:
    st.header('💬 Conversations')
    
    if st.button("➕ New Chat", use_container_width=True):

        cid = str(uuid.uuid4())

        st.session_state.conversations[cid] = {
            "title": "New Chat",
            "messages": []
        }

        st.session_state.history_store[cid] = ChatMessageHistory()

        st.session_state.current_chat_id = cid

        save_conversations()

        st.rerun()

    for cid, chat in list(st.session_state.conversations.items()):
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(chat['title'], key='open_' + cid, use_container_width=True):
                st.session_state.current_chat_id = cid
                st.rerun()
        with col2:
            if st.button('🗑️', key='del_' + cid):

                st.session_state.conversations.pop(cid, None)

                st.session_state.history_store.pop(cid, None)

                save_conversations()

                if st.session_state.current_chat_id == cid:

                    if st.session_state.conversations:

                        st.session_state.current_chat_id = list(
                            st.session_state.conversations.keys()
                        )[0]

                    else:

                        new_cid = str(uuid.uuid4())

                        st.session_state.conversations[new_cid] = {
                            'title': 'New Chat',
                            'messages': []
                        }

                        st.session_state.history_store[new_cid] = ChatMessageHistory()

                        st.session_state.current_chat_id = new_cid

                        save_conversations()

                st.rerun()

# Display current chat
current_chat = st.session_state.conversations[st.session_state.current_chat_id]

for msg in current_chat['messages']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Handle user input
query = st.chat_input('Ask anything about Aras...')

if query:
    # Update chat title if it's a new chat
    if current_chat['title'] == 'New Chat':
        current_chat['title'] = query[:35] + ('...' if len(query) > 35 else '')

    # Add user message
    current_chat['messages'].append({'role': 'user', 'content': query})
    save_conversations()
    
    with st.chat_message('user'):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = chat_chain.invoke(
                    {"input": query},
                    config={
                        "configurable": {
                            "session_id": st.session_state.current_chat_id
                        }
                    }
                )

                answer = result["answer"]
                st.markdown(answer)

                # Collect and display sources
                sources = []
                for doc in result.get("context", []):
                    src = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page = doc.metadata.get("page_number") or doc.metadata.get("page")
                    
                    if page is not None:
                        src = f"{src} (Page {page})"
                    
                    sources.append(src)

                # Remove duplicates while preserving order
                sources = list(dict.fromkeys(sources))

                # Display sources
                if sources:
                    with st.expander("📄 Sources"):
                        for src in sources:
                            st.write(f"- {src}")
                # Save assistant response
                current_chat['messages'].append({'role': 'assistant', 'content': answer})
                save_conversations()
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                logger.error(f"Error processing query: {e}", exc_info=True)
                current_chat['messages'].append({'role': 'assistant', 'content': error_msg})

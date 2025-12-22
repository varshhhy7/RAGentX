import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import PINECONE_API_KEY

# ---------------------------------------------------------
# 1. Environment & Client Setup
# ---------------------------------------------------------

# Make API key available to all libraries that expect it
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone client (connection to vector DB service)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize embedding model (text → vector)
# IMPORTANT: Same model must be used for both storing & querying
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Name of the Pinecone index (vector table)
INDEX_NAME = "rag-index"


# ---------------------------------------------------------
# 2. Create or Load Vector Store (Retriever)
# ---------------------------------------------------------

def get_vector_store():
    """
    Ensures Pinecone index exists and returns a retriever object.

    Retriever = abstraction used by LangChain to fetch
    relevant documents based on similarity search.
    """

    existing_indexes = [index["name"] for index in pc.list_indexes()]

    # Create index only if it doesn't already exist
    if INDEX_NAME not in existing_indexes:
        print("Creating new Pinecone index...")

        pc.create_index(
            name=INDEX_NAME,
            dimension=embeddings.dimension,  # must match embedding size
            metric="cosine",                 # cosine similarity is standard for text
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print(f"Created Pinecone Index: {INDEX_NAME}")

    # Connect LangChain to the Pinecone index
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    # Retriever is what RAG pipelines actually use
    return vectorstore.as_retriever()


# ---------------------------------------------------------
# 3. Upload Documents to Pinecone
# ---------------------------------------------------------

def upload_docs(text_content: str):
    """
    Splits raw text into chunks, embeds them,
    and uploads them to Pinecone.
    """

    if not text_content.strip():
        raise ValueError("Document content cannot be empty")

    # Splits large text into overlapping chunks
    # chunk_overlap helps preserve context across chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    print("Splitting document into chunks...")
    documents = text_splitter.create_documents([text_content])

    # Connect to existing vector store
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    print(f"Uploading {len(documents)} chunks to Pinecone...")
    vectorstore.add_documents(documents)

    print("Successfully added document chunks to Pinecone.")

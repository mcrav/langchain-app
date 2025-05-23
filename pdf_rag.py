from openrouter import ChatOpenRouter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains import RetrievalQA
import sys


model = ChatOpenRouter(model_name="meta-llama/llama-3.3-8b-instruct:free")


def main():
    file_path = sys.argv[1]

    # Load PDF and split into chunks
    loader = PyPDFLoader(file_path)

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    # Create vector store of PDF chunks
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = InMemoryVectorStore(embeddings)

    vector_store.add_documents(documents=all_splits)

    # Create retriever and QA chain
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    qa_chain = RetrievalQA.from_llm(model, retriever=retriever)

    print(qa_chain(sys.argv[2]))


if __name__ == "__main__":
    main()

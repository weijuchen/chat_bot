# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

from langchain_openai import OpenAIEmbeddings

from langchain.chains import RetrievalQA, create_retrieval_chain

# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

# from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import ConversationalRetrievalChain
import os

# import sys

# print(sys.path)


# *** Created function to load multiple PDFs
pdf_folder_path = "/chat_bot/06afterclean"

# pdf_folder_path = r"D:\Computer Science\AI\LLM model\chat_bot\06afterclean"
def load_multiple_pdfs(pdf_folder_path):  # ***
    docs = []  # *** Initialize empty list to store documents
    # *** Loop through all files in the specified folder
    for file_name in os.listdir(pdf_folder_path):  # ***
        if file_name.endswith(".pdf"):  # *** Check if file is a PDF
            file_path = os.path.join(
                pdf_folder_path, file_name
            )  # *** Get full path of the file
            loader = PyPDFLoader(file_path)  # *** Load the PDF using PyPDFLoader
            docs.extend(loader.load())  # *** Add loaded documents to docs list
    return docs  # *** Return the full list of documents


openai_api_key = os.getenv("OPENAI_API_KEY")
def create_vector():
    pdf_folder_path = "/chat_bot/06afterclean"
    # pdf_folder_path=r"D:\Computer Science\AI\LLM model\chat_bot\06afterclean"
    docs = load_multiple_pdfs(pdf_folder_path)

    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=openai_api_key,  # Pass your OpenAI API Key here
    )

    # split your docs into texts chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""],
    )
    texts = text_splitter.split_documents(docs)

    # embed the chunks into vectorstore (FAISS)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    vectorstore.save_local("faiss_midjourney_docs")
    # print("vectorstore created")
    return vectorstore


# # *** Call the qa_agent function, passing in the API Key 我剛在網路認識一個人，他說他是醫生，但是有急用然後要向我借錢，我想知道這是詐騙嗎？and folder path for PDFs

# get_qa_chain(
#     openai_api_key=openai_api_key,
#     question="我剛在網路認識一個人，他說他是醫生，但是有急用然後要向我借錢，我想知道這是詐騙嗎？ ",
# )

# question_answer_chain = create_stuff_documents_chain(llm=ChatOpenAI(model_name='gpt-4o-mini', temperature=0),
#    prompt=rag_prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# print(f"Type of rag_chain: {type(rag_chain)}")
# print("RAG chain created"+type(rag_chain))
# return rag_chain


# rag_chain = get_qa_chain(openai_api_key
# ,r"D:\Computer Science\AI\LLM model\chat_bot\06afterclean")


# result = rag_chain({"input": "What is the capital of France?"})
# print(result['answer'])

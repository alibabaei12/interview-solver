from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

if __name__ == '__main__':
    pdf_path = '/Users/alibabaei/Code/git/interview-solver/Cracking-the-Coding-Interview-6th-Edition-189-Programming-Questions-and-Solutions.pdf'

    loader = PyPDFLoader(file_path=pdf_path)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=document)

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_prompt")

    new_vectorstore = FAISS.load_local("faiss_index_prompt", embeddings)
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0,model_name="gpt-4"), chain_type="stuff", retriever=new_vectorstore.as_retriever())

    question = """
Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

The overall run time complexity should be O(log (m+n)).

 
"""
    prompt = f""" 
    Pretend that you are very good at doing software engineering interview questions. I have provided detailed programming concepts and examples in a PDF document. The PDF does not contain the answer to the question directly but provides algorithms and teaches how to approach the question. Use your knowledge and the context from the PDF to give the final answer.

Question: {question}

Before you answer, briefly state which algorithm and data structure you will use to solve the question. Then, provide a step-by-step explanation of the solution. Each step should be clear, concise, and informative, containing no more than 25 words. The goal is to convey the solution process clearly and succinctly, so it can be understood within 1 minute.

Finally, please write the code as well in Python 3, ensuring it's clean and efficient.

Answer using both the PDF document and your general knowledge, and give clear answers in bullet points:
   """

    res = qa.run(prompt)
    print(res)

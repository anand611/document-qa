
# from openai import OpenAI
from langchain_mistralai import ChatMistralAI
# from mistralai import Mistral
import os
from langchain_mistralai import MistralAIEmbeddings

# from langchain.llms import MistralAI

from langchain.chains import RetrievalQA
# from langchain.llms import MistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.vectorstores import VectorStoreRetriever

model = "ministral-3b-latest"

def get_embeddings_model(model="mistral-embed",api_key=""):
    if model and api_key:
        embeddings = MistralAIEmbeddings(model=model,api_key=api_key)
        vectorstore = InMemoryVectorStore(embedding=embeddings)
        return embeddings,vectorstore
    else:
        return None
def push_to_vectorstore(vectorstore:InMemoryVectorStore,splitted_text):
    if vectorstore and splitted_text:
        ids = vectorstore.add_documents(documents=splitted_text)
        return vectorstore,ids        
        
# Show title and description.
st.title("📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    # "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "To use this app, you need to provide an MistralAI API key"
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
# openai_api_key = st.text_input("OpenAI API Key", type="password")
mistralai_api_key = st.text_input("MistralAI API Key", type="password")
if not mistralai_api_key:
    st.info("Please add your MistralAI API key to continue.", icon="🗝️")
else:
    os.environ["MISTRAL_API_KEY"] = mistralai_api_key
    # Create an OpenAI client.
    # client = OpenAI(api_key=openai_api_key)

    # Create a mistralAI client.
    # llm = ChatMistralAI(
    #    model=model,
    #    mistral_api_key=mistralai_api_key,
        # streaming=True
        # temperature=0,
        # max_retries=2,
        # other params...
    # )
    # client = Mistral(api_key=mistralai_api_key)

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.pdf/.txt/.md)", type=("pdf","txt", "md")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        # Step 2: Load and Process the PDF
        temp_file = "./temp.pdf"
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
        loader = PyPDFLoader(temp_file)
        docs = loader.load()
        
        # Step 3: Split Text into Chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,add_start_index=True)
        chunks = text_splitter.split_documents(documents=docs)
        
        # Step 4: Create a Vector Store
        embeddings = MistralAIEmbeddings(model="mistral-embed",api_key=mistralai_api_key)        
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Step 5: Set Up the Retriever
        retriever = vector_store.as_retriever()

        # Step 6: Build the Question/Answering System
        llm = ChatMistralAI(model=model)
        qa_chain = RetrievalQA(llm=llm, retriever=retriever)

        response = qa_chain.invoke(query)
        st.write(response)
        
        # st.markdown(documents)
        #######################################################################################################
        system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use three sentences maximum and keep the answer concise. "
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # question_answer_chain = create_stuff_documents_chain(llm, prompt)
        #######################################################################################################
        
        # Create embeddings for the documents
        # embeddings = MistralAIEmbeddings(model="mistral-embed",api_key=mistralai_api_key)        
        # embeddings,vectorstore = get_embeddings_model(api_key=mistralai_api_key)

        #push data to vector store
        # vctr_str,idx=push_to_vectorstore(vectorstore=vectorstore,splitted_text=splitted_text)
        
        # creating vectors
        # content_vectors = embeddings.embed_query(splitted_text)
        # creating memory vector store.
        # vectorstore = InMemoryVectorStore(embedding=embeddings)
        # adding documents to vector store
        # vectorstore.add_documents(documents=splitted_text)

        # Performing the search in the vector store
        # response = vectorstore.similarity_search(query=question)

        # retriever = vctr_str.as_retriever()

        # chain=create_retrieval_chain(retriever,question_answer_chain)
        # retriever = VectorStoreRetriever(vectorstore=vctr_str)
        # retrievalQA = RetrievalQA.from_llm(llm=llm,retriever = retriever)
        # st.write(retrievalQA.invoke({"input":question}))
        # chain.invoke({"input":question})
        
        # Initialize the language model
        # llm = ChatMistralAI(model="mistral-7b")
        
        # Create the RetrievalQA chain
        # qa_chain = RetrievalQA(llm=llm, retriever=vectorstore.as_retriever())

        # Ask a question
        # query = "What is the main topic of the document?"
        # response = qa_chain.ainvoke(query)
        
        # Process the uploaded file and question.
        # document = uploaded_file.read().decode()
        # messages = [
        #     {
        #        "role": "user",
        #        "content": f"Here's a document: {document} \n\n---\n\n {question}",
        #    }
        # ]

        # Generate an answer using the OpenAI API.
        # stream = client.chat.completions.create(
        #    model="gpt-3.5-turbo",
        #    messages=messages,
        #    stream=True,
        #)

        # response = llm.invoke(messages)
        
        # print(response)
        # st.write(response.choices[0].message.content)
        
        # Stream the response to the app using `st.write_stream`.
        # st.write_stream(stream)
        # for chunk in stream_response:
        #    print(chunk.data.choices[0].delta.content)

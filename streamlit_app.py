import streamlit as st
# from openai import OpenAI
from langchain_mistralai import ChatMistralAI
# from mistralai import Mistral
import os
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
# from langchain.llms import MistralAI
from langchain.vectorstores import faiss

model = "ministral-3b-latest"
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

    # Create an mistralAI client.
    llm = ChatMistralAI(
        model=model,
        mistral_api_key=mistralai_api_key,
        # streaming=True
        # temperature=0,
        # max_retries=2,
        # other params...
    )
    # client = Mistral(api_key=mistralai_api_key)

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("pdf","txt", "md")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        temp_file = "./temp.pdf"
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
        loader = PyPDFLoader(temp_file)
        documents = loader.load()

        # Create embeddings for the documents
        embeddings = MistralAIEmbeddings(model="mistral-embed",api_key=mistralai_api_key)
        vectorstore = faiss.FAISS.from_documents(documents, embeddings)

        # Initialize the language model
        # llm = MistralAI(model="mistral-7b")
        
        # Create the RetrievalQA chain
        qa_chain = RetrievalQA(llm=llm, retriever=vectorstore.as_retriever())

        # Ask a question
        query = "What is the main topic of the document?"
        response = qa_chain.run(query)
        
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
        print(response)
        # st.write(response.choices[0].message.content)
        
        # Stream the response to the app using `st.write_stream`.
        # st.write_stream(stream)
        # for chunk in stream_response:
        #    print(chunk.data.choices[0].delta.content)

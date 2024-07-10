import streamlit as st
import tiktoken
from loguru import logger
import os 
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser
import pandas as pd
import koreanize_matplotlib

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

def main():
    st.set_page_config(
        page_title = "SURVEY GO",
        page_icon = "📋"
    )

    st.title(" 📋 SURVEYGO CHATBOT")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None


    class StreamlitResponse(ResponseParser):
        def __init__(self, context) -> None:
            super().__init__(context)

        def format_dataframe(self, result):
            st.dataframe(result['value'])
            return
        def format_plot(self, result):
            st.image(result['value'])
            return
        def format_other(self, result):
            st.write(result['value'])
            return

    # option = st.selectbox(
    # "사용할 데이터의 종류를 선택해주세요.",
    # ("PDF", "CSV", "DOCS"),
    # index=None,
    # placeholder="Select contact method...",
    # )

    option = st.selectbox("사용할 데이터의 종류를 선택하신 후 왼쪽에서 데이터를 로드해주세요.",("PDF", "DOCS","CSV"),index=None,placeholder="Select contact method...",)
    if option == 'CSV':
        with st.sidebar:
            uploaded_file = st.file_uploader("Upload your file",type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write(df.head(15))
            process = st.button("데이터 사용하기")
            if process:
                st.write("데이터가 적용되었습니다.")


        query = st.text_area("🔎 질문을 입력하세요 ")
        if query:
            llm = OpenAI(api_token=os.environ["OPENAI_API_KEY"])
            query_engine = SmartDataframe(df,
                                        config ={
                                            "llm": llm,
                                                "response_parser": StreamlitResponse,
                                                },
                                                )
            answer = query_engine.chat(query)
            st.write(answer)
    else:
        with st.sidebar:
            uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'])
            process = st.button("Process")
        if process:
            files_text = get_text(uploaded_files)
            text_chunks = get_text_chunks(files_text)
            vetorestore = get_vectorstore(text_chunks)
        
            st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 

            st.session_state.processComplete = True

        if 'messages' not in st.session_state:
            st.session_state['messages'] = [{"role": "assistant", 
                                            "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        history = StreamlitChatMessageHistory(key="chat_messages")

        # Chat logic
        if query := st.chat_input("질문을 입력해주세요."):
            st.session_state.messages.append({"role": "user", "content": query})

            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                chain = st.session_state.conversation

                with st.spinner("Thinking..."):
                    result = chain({"question": query})
                    with get_openai_callback() as cb:
                        st.session_state.chat_history = result['chat_history']
                    response = result['answer']
                    source_documents = result['source_documents']

                    st.markdown(response)
                    with st.expander("참고 문서 확인"):
                        st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                        st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                        st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                        


    # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(doc):
    doc_list = []
    file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
    with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
        file.write(doc.getvalue())
        logger.info(f"Uploaded {file_name}")
    if '.pdf' in doc.name:
        loader = PyPDFLoader(file_name)
        documents = loader.load_and_split()
    elif '.docx' in doc.name:
        loader = Docx2txtLoader(file_name)
        documents = loader.load_and_split()
    elif '.pptx' in doc.name:
        loader = UnstructuredPowerPointLoader(file_name)
        documents = loader.load_and_split()

    doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain


if __name__ == '__main__':
    main()

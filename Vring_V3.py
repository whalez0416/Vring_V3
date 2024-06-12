#콜백핸들러 오류 해결 필요 
#메시지 로그 사라지지 않는 문제 해결 필요

import os ## 운영체제와 연동하기 위한 모듈
import json ## 메시지를 json 형식으로 저장하기 위한 모듈
import streamlit as st #스트림릿 연결
from langchain.prompts import ChatPromptTemplate #프롬프트를 통한 llm instruction
from langchain.document_loaders import UnstructuredFileLoader # pdf, txt, word 등 다양한 파일 등록 가능하도록 document_loaders 설치
from langchain.embeddings import OpenAIEmbeddings #오픈 ai 임베딩
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough ###
from langchain.text_splitter import RecursiveCharacterTextSplitter #재귀적 캐릭터텍스트스플릿터, 반점, 온점 등을 기준으로 끊음
from langchain.vectorstores.faiss import FAISS #백터 스토어
from langchain.callbacks.base import BaseCallbackHandler ### 콜백핸들러 모듈
from langchain.chat_models import ChatOpenAI #llm
from langchain.retrievers.multi_query import MultiQueryRetriever #리트리버
from google.cloud import storage #캐쉬를 클라우드에도 저장되도록 구글 클라우드 연결 
from google.oauth2 import service_account #구글 클라우드 서비스 인증처리
import pickle # 저장되는 데이터를 보다 빠르게 불러오고, 복잡한 데이터를 효율적으로 저장하기 위한 모듈
import tempfile #데이터를 로컬에 임시 저장하기 위한 모듈

# Streamlit secrets 에 toml 파일로 구글 키가 업데이트가 안되니 구글 클라우드 키를 secrets서 읽을수 있도록 하는 키 
service_account_info = st.secrets["gcp_service_account"]
credentials = service_account.Credentials.from_service_account_info(service_account_info)
storage_client = storage.Client(credentials=credentials)


# Streamlit 페이지 설정
st.set_page_config(page_title="Vring_V3", page_icon="📃")

# Google Cloud Storage에 파일 업로드하는 함수 버켓 이름, 로컬에 있는 업로드할 파일 이름, gcs에 저장할 blob 을 받음  
def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    try: 
        #버켓을 호출
        bucket = storage_client.bucket(bucket_name)
        #블랍은 대규모 이진 데이터 저장하는 것 앞서 정의한 버켓을 이진 데이터로 저장
        blob = bucket.blob(destination_blob_name)
        #저장한 이진 데이터를 앞서 만든 파일 네임과 동일하게 업로드
        blob.upload_from_filename(source_file_name)
        #파일이이 업로드 되면 해당 문구를 출력함 "{로컬파일이름}이 {버켓이름}/{이진파일이름}에 저장되었다"
        print(f"{source_file_name} has been uploaded to {bucket_name}/{destination_blob_name}")
    #예외가 발생했을때는 업로드에 실패했다는 문구를 출력
    except Exception as e:
        print(f"Failed to upload {source_file_name} to GCS: {e}")
        
        
        
# 메시지 저장 함수: 카카오톡 처럼 대화가 계속 누적되며 저장될 수 있도록 하기 위한 함수
def save_message(message, role): 
    #스트림릿 세션 스테이트(특정 상태를 저장할 수 있는 상태) 안에 아무런 메시지가 없다면 빈 리스트를 출력
    if "messages" not in st.session_state:
        #여기서 대괄호는 딕셔너리와 같이 이미 저장된 messages를 출력하는 것이고 이미 저장된 메시지가 없다면 빈 리스트를 출력하라는 뜻 
        st.session_state["messages"] = [] 
    #앞서 출력된 빈리스트에 메시지와 role을 지속적으로 세션스테이트에 추가되도록 설정
    st.session_state["messages"].append({"message": message, "role": role}) 
    #이 대화 기록을 json 파일로 저장하도록 설정 messages.json은 파일 이름 "w"는 쓰기 모드(쓰기 모드는 기존 파일이 있으면 내용을 덮어쓰고, 파일이 없으면 새로 만듬.) 이걸 as f로 받음
    with open("messages.json", "w") as f:  
        #dump는 리스트인 파이썬 객체를 json 파일로 변환 하는데 여기서는 세션스테이트에 저장된 messages를 변환하고 앞에 정의한f 에다가 기록
        json.dump(st.session_state["messages"], f) 
    #그리고 그걸 앞서 정의한 구글 클라우드 업로드 함수로 업로드 함 
    upload_to_gcs('goldgpt_v2', 'messages.json', 'messages.json')

#콜백 핸들러: 프로그램이 작동되는 상황을 세분화하고 그 과정마다 특정 명령을 내릴 수 있는 모듈
class ChatCallbackHandler(BaseCallbackHandler):
    #이 클래스 아래 포함되는 모든 메소드에 영향을 끼치는 속성 값으로 모든 메시지는 빈 문자열로 출력하도록 설정
    message = ""
    #llm이 시작할때 메시지 박스에 st.empty로 빈 요소를 만듬
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
        self.message = ""  # 메시지 초기화
    #llm이 종료될때 save_message 함수로 메시지를 json 형식으로 저장ㄴ
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
    #토큰이 추가될때마다 마크다운 형식으로 빈 문자열인 메시지에 채워지도록 설정하여 스트리밍 형태로 보여지도록 함 
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


# LLM 설정
llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler(),],
)



# Google Cloud Storage에서 파일 다운로드 이걸 왜하는거지 ?
def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"{source_blob_name} has been downloaded to {destination_file_name}")
    except Exception as e:
        print(f"Failed to download {source_blob_name} from GCS: {e}")

# Google Cloud Storage에서 파일 다운로드하여 임시 파일로 로드(유저가 업로드 한 파일은 임베딩후 업로드가 되며, 업로드 된 파일은 로컬에도 자동저장되어야함)
def load_from_gcs(bucket_name, source_blob_name):
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        temp_local_filename = tempfile.mkstemp()
        blob.download_to_filename(temp_local_filename)
        with open(temp_local_filename, "rb") as f:
            data = pickle.load(f)
        os.remove(temp_local_filename)
        return data
    except Exception as e:
        print(f"Failed to load {source_blob_name} from GCS: {e}")
        return None

# Google Cloud Storage에 파일 업로드하여 저장
def save_to_gcs(bucket_name, destination_blob_name, data):
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        _, temp_local_filename = tempfile.mkstemp()
        with open(temp_local_filename, "wb") as f:
            pickle.dump(data, f)
        blob.upload_from_filename(temp_local_filename)
        os.remove(temp_local_filename)
        print(f"{destination_blob_name} has been saved to {bucket_name}")
    except Exception as e:
        print(f"Failed to save {destination_blob_name} to GCS: {e}")

# 파일 임베딩 및 캐싱 함수
def embed_file(file):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.name)
    file_content = file.read()
    
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    bucket_name = 'goldgpt_v2'  # GCS 버킷 이름
    upload_to_gcs(bucket_name, file_path, file.name)

    cache_file_path = f"embeddings/{file.name}.pkl"
    
    multiquery_retriever = None  # 초기화

    # 캐시 파일 존재 여부 확인
    vectorstore = load_from_gcs(bucket_name, cache_file_path)
    if vectorstore is None:
        try:
            st.write("Creating new embeddings...")
            loader = UnstructuredFileLoader(file_path)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
            )
            docs = loader.load_and_split(text_splitter=splitter)
            st.write(f"Loaded and split {len(docs)} documents.")
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)
            st.write("Created FAISS vectorstore.")
            
            # FAISS 인덱스 저장
            save_to_gcs(bucket_name, cache_file_path, vectorstore)
            
            multiquery_retriever = MultiQueryRetriever.from_llm(
                retriever=vectorstore.as_retriever(), llm=llm
            )
            st.write("Created MultiQueryRetriever.")
        except Exception as e:
            st.error(f"Failed to create embeddings: {e}")
            print(f"Failed to create embeddings: {e}")
    else:
        try:
            st.write("Loaded cached embeddings...")
            multiquery_retriever = MultiQueryRetriever.from_llm(
                retriever=vectorstore.as_retriever(), llm=llm
            )
            st.write("Loaded MultiQueryRetriever from cache.")
        except Exception as e:
            st.error(f"Failed to load cached embeddings: {e}")
            print(f"Failed to load cached embeddings: {e}")

    if multiquery_retriever is None:
        raise ValueError("Failed to create or load embeddings.")
    
    return multiquery_retriever


# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", """
        you are the head of professional consultation at the urology department. Our goal is to attract patients to our hospital by responding to all inquiries with the utmost kindness and warmth. you provide answers based only on the given context. If you are uncertain about a query, you kindly and warmly encourage the inquirer to contact the hospital directly for more information..
        Additionally, to make patients feel more at ease, I use emoticons like the following:
        Example 1: Hello~ Good morning!
        Example 2: Thank you for your question ^^ The answer is as follows!
        If I have already 안녕하세요, I will not repeat 안녕하세요.
        Context: {context}
    """),
    ("human", "{question}"),
])

# Streamlit UI 설정
st.title("Vring_V3")
st.markdown("""
궁금한 내용을 챗봇에게 물어보시면 답변드리겠습니다.
""")





# 메시지 보내기
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)
        
# 파일 업로드 처리
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )
    
    
# 문서 포맷팅
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

# 파일이 업로드되거나 삭제될 때 세션 상태 초기화
if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)


else:
    st.session_state["messages"] = [] 
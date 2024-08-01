import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

import textwrap
import time


load_dotenv()

# Helper function to format text in Markdown
def to_markdown(text):
    text = text.replace('•', '  *')
    return "> " + textwrap.indent(text, '> ', predicate=lambda _: True).replace('\n', '\n> ')






# Define the embeddings and LLM models
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

# Create a FAISS database from a YouTube video URL
def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=3):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that can answer questions about YouTube videos
        based on the video's transcript.

        Answer the following question: {question}
        By searching the following video transcript: {docs}

        Only use the factual information from the transcript to answer the question.

        If you feel like you don't have enough information to answer the question, say "I don't know".

        Your answers should be verbose and detailed.
        """,
    )

    chain = prompt | llm
    response = chain.invoke(
        {
            "question": query,
            "docs": docs_page_content
        }
    )

    return response.content, docs


def response_generator(db, QUERY):
    response, docs = get_response_from_query(db, QUERY)
    
    for word in response.split():
        yield word + " "
        time.sleep(0.06)

    st.markdown("**Top 3 Related Transcript Sections:**")
    for doc in docs:
        st.markdown(to_markdown(doc.page_content))


st.title('🧐Search in YouTube Video ✨')

YOUTUBE_URL = st.text_input("Enter YouTube URL", "https://youtu.be/pJ0auP7dbcY?si=-1W_lYG6V0lygcmw")
st.video(YOUTUBE_URL)

QUERY = st.text_input("Enter your query", "What did Dr. Yasser say about authority? Explain")

if st.button("Get Response"):
    with st.spinner('Processing...'):
        db = create_db_from_youtube_video_url(YOUTUBE_URL)
        response = response_generator(db, QUERY)
        to_markdown(st.write_stream(response))



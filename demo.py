from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document 
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
import numpy as np
import os
import re

ut_api=YouTubeTranscriptApi() 

def extract_video_id(url):
    pattern=r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/|youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})'
    
    match=re.search(pattern,url)
    if match:
        return match.group(1)
    return None

user_input = input("Paste your YouTube URL: ")
video_id = extract_video_id(user_input)

if not video_id:
    print("Error: That doesn't look like a valid YouTube link!")
else:
    print(f"Success! Processing Video ID: {video_id}")

try: 
    transcript_fetch=ut_api.fetch(video_id,languages=['en','hi']) 
    transcript=" ".join(chunk.text for chunk in transcript_fetch) 
except TranscriptsDisabled: 
    print("No captions available for this video.") 
    
fetch_list=[] 
for details in transcript_fetch: 
    text=details.text 
    start=details.start 
    end=details.start+details.duration 
    fetch_list.append( 
                    {'text':text.strip(), 
                    'start':start, 
                    'end':end 
                })
        
merged_docs = [] 
chunk_size = 40
step_size = 20 
for i in range(0, len(fetch_list), step_size): 
    chunk = fetch_list[i:i+chunk_size] 
    if not chunk: 
        continue 
    text = " ".join(item['text'] for item in chunk) 
            
    start = chunk[0]['start'] 
    end = chunk[-1]['end'] 
    merged_docs.append({ 'text': text, 
                        'start': start, 
                        'end': end })
    
docs=[] 
for item in merged_docs:
    doc=Document( 
                 page_content= item['text'],
                 metadata={ 'start':item['start'], 
                           'end':item['end'] } ) 
    docs.append(doc) 
                
splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100) 
chunks = splitter.split_documents(docs)

embedding= OllamaEmbeddings(model='nomic-embed-text-v2-moe:latest')

FAISS_INDEX_DIR = "faiss_index"
if os.path.exists(FAISS_INDEX_DIR):
    vector_store = FAISS.load_local(
        FAISS_INDEX_DIR,
        embedding,
        allow_dangerous_deserialization=True
    )
else:
    vector_store = FAISS.from_documents(chunks, embedding)
    vector_store.save_local(FAISS_INDEX_DIR)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})

llm = ChatOllama(model="mistral") 

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use only the provided context to answer.If the answer is not in the context, say I don't know."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])
chat_history = []

try:
    with open("chat_history.txt", "r") as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()

            if line.startswith("Human:"):
                content = line.replace("Human:", "").strip()
                chat_history.append(HumanMessage(content=content))

            elif line.startswith("AI:"):
                content = line.replace("AI:", "").strip()
                chat_history.append(AIMessage(content=content))

except FileNotFoundError:
    print("No previous chat history found. Starting fresh.")
while True:
    question = input("You: ")

    if question.lower() in ["exit", "quit"]:
        print("Chat ended.")
        break

    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = chat_template.invoke({
        "chat_history": chat_history,
        "context": context_text,
        "question": question
    })

    response = llm.invoke(final_prompt)
    print("Bot:", response.content)
    
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response.content))
    
    with open("chat_history.txt", "a") as f:
        f.write(f"Human: {question}\n")
        f.write(f"AI: {response.content}\n")



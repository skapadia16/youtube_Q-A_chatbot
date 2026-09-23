import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace,HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

##indexing(document ingestion)

video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    ytt_api = YouTubeTranscriptApi()
    # If you don’t care which language, this returns the “best” one
    transcript = ytt_api.fetch(video_id, languages=["en"])

    # Flatten it to plain text
    transcript_text = " ".join(
        snippet.text for snippet in transcript
    )

    # print(transcript_text)

except TranscriptsDisabled:
    print("Transcript is disabled for this video.")

## indexiing(text splitting)

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap= 200)
chunks = splitter.create_documents([transcript_text])

# print(len(chunks))

##indexing (embedding generation and storing in vector store)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(chunks, embeddings)

# print(vector_store.index_to_docstore_id)

##retrieval

retriever = vector_store.as_retriever(search_type = "similarity",search_kwargs={"k":4})

print(retriever.invoke("what is deepmind"))


## augmentation

client = InferenceClient(
    token=os.getenv("HF_TOKEN")
)

question = input("Enter question related to video")

# response = client.chat.completions.create(
#     model="openai/gpt-oss-120b",
#     messages=[
#         {
#             "role": "user",
#             "content": question
#         }
#     ],
#     temperature=0.2,
#     max_tokens=512
# )

# print(response.choices[0].message.content)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      context : {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

# question = "is the topic of nuclear fusion discussed in this video ? if yes what is that "
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
# print(context_text)

final_prompt = prompt.invoke({"context": context_text, "question": question})
# print(final_prompt)

## generation 
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": final_prompt.to_string()
        }
    ],
    temperature=0.2,
    max_tokens=512
)

print(response.choices[0].message.content)

# def format_docs(retrieved_docs):
#   context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
#   return context_text

# parallel_chain = RunnableParallel({
#     'context': retriever | RunnableLambda(format_docs),
#     'question': RunnablePassthrough()
# })

# parallel_chain.invoke('who is Demis')

# parser = StrOutputParser()

# main_chain = parallel_chain | prompt | response | parser

# main_chain.invoke("can you summarize the video ")
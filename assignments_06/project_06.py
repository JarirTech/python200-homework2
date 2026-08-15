from dotenv import load_dotenv
import os
from pathlib import Path
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")



from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)

from llama_index.llms.openai import OpenAI

from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)

#Step 1: Setup
print("Step 1 ***********************************************************")

docs_dir = Path(r"C:\Users\bjari\OneDrive\Desktop\python-200\lessons\06_AI_augmentation\resources\groundwork_docs")

assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

########################################################################

#Step 2: Load the Documents

print("Step 2 ***********************************************************")


documents = SimpleDirectoryReader(
    str(docs_dir)
).load_data()

print(f"\nLoaded {len(documents)} documents:\n")

for doc in documents:
    print(doc.metadata["file_name"])

####################################################################
#Step 3: Build the Index and Query Engine
print("Step 3 ***********************************************************")
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

if index:
    print('Index built successfully')

##################################################################

#Step 4: Query the Assistant
print("Step 4 ***********************************************************")
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

# #The question
# The answer from the model
# The top retrieved source node
for q in questions:
    print("Question: ", q)

    response = query_engine.query(q)
    print(response)
    print("\nRetrieved Source Nodes:")

    for i, node in enumerate(response.source_nodes, start=1):
        print(f"\nNode {i}")
        print(f"Score: {node.score}")
        print(f"Text Preview: {node.text[:150]}")

# #The assistant gave mostly accurate and confident answers.
# Some retrieved chunks were very relevant, while others were only loosely related.
# Even when retrieval was imperfect, the model was still able to answer correctly in most cases.
#Step 5: Find a Failure
print("Step 5 ***********************************************************")
failure_query = ("Is Groundwork facing a bunkcrupcy?")
failure_response = query_engine.query(failure_query)
print("\nQUESTION:")
print(failure_query)

print("\nANSWER:")
print(failure_response)

print("\nRetrieved Source Nodes:")

for i, node in enumerate(failure_response.source_nodes, start=1):

    print(f"\nNode {i}")

    print(
        f"Document: "
        f"{node.metadata.get('file_name', 'Unknown')}"
    )

    print(f"Score: {node.score}")

# I asked if Groundwork was facing bankruptcy because the documents
# did not contain any financial information.

# The retrieval returned loosely related documents like the company
# story and menu, but nothing about finances.

# The model answered carefully and said there was no information
# about bankruptcy instead of making up details.

# This shows that RAG systems can still give reasonable answers
# even when the information is missing, but the responses should
# still be verified.
print("Step 6 ***********************************************************")

# Final reflection
# Using LlamaIndex made building the RAG system much easier and shorter
# than creating the whole semantic pipeline manually. The framework handled
# chunking, embeddings, and retrieval with only a few lines of code.

# Another useful use case would be a hospital assistant that answers
# questions from internal policies and patient procedure documents.

# One failure mode RAG cannot fully prevent is hallucination. Even when
# retrieval works correctly, the model can still misunderstand or give
# an incomplete answer.
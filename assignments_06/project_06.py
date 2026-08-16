# Part 2: Mini-Project - Groundwork Coffee Co. Q&A Assistant

from dotenv import load_dotenv
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding



# Step 1: Setup


print("Step 1 ***********************************************************")

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# Path to the Groundwork documents
docs_dir = Path(
    r"C:\Users\bjari\OneDrive\Desktop\python-200"
    r"\lessons\06_AI_augmentation\resources\groundwork_docs"
)

# Make sure the document folder exists
assert docs_dir.exists(), (
    f"Document directory not found: {docs_dir}"
)

# Configure the LLM and embedding model
Settings.llm = OpenAI(model="gpt-4o-mini")

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)



# Step 2: Load the Documents


print("Step 2 ***********************************************************")

documents = SimpleDirectoryReader(
    str(docs_dir)
).load_data()

print(f"\nLoaded {len(documents)} documents:")

# Print the filename of every document
for doc in documents:
    print(f"- {doc.metadata.get('file_name', 'Unknown')}")



# Step 3: Build the Index and Query Engine


print("\nStep 3 ***********************************************************")

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(
    similarity_top_k=3
)

print("Index built successfully. Ready to answer questions.")



# Step 4: Query the Assistant


print("\nStep 4 ***********************************************************")

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:

    print("\n" + "=" * 70)
    print("QUESTION:")
    print(q)

    response = query_engine.query(q)

    print("\nANSWER:")
    print(response)

    print("\nTOP RETRIEVED SOURCE NODE:")

    # Print the top retrieved source node
    if response.source_nodes:
        node = response.source_nodes[0]

        print(
            f"Document: "
            f"{node.metadata.get('file_name', 'Unknown')}"
        )

        print(f"Similarity Score: {node.score}")

        print(
            f"Text Preview: "
            f"{node.text[:200]}"
        )
    else:
        print("No source nodes were retrieved.")


# Step 4 Reflection:
#
# The assistant gave mostly accurate answers.
# Some answers had the correct information, even when the top
# retrieved document was not the best match. This shows that
# retrieval does not always find the most relevant document.


# Step 5: Find a Failure


print("\nStep 5 ***********************************************************")

failure_query = "Is Groundwork facing bankruptcy?"

failure_response = query_engine.query(
    failure_query
)

print("\nQUESTION:")
print(failure_query)

print("\nFULL RESPONSE:")
print(failure_response)

print("\nALL THREE RETRIEVED SOURCE NODES:")

for i, node in enumerate(
    failure_response.source_nodes,
    start=1
):

    print(f"\nNode {i}")

    print(
        f"Document: "
        f"{node.metadata.get('file_name', 'Unknown')}"
    )

    print(
        f"Similarity Score: "
        f"{node.score}"
    )

    print(
        f"Text Preview: "
        f"{node.text[:200]}"
    )


# Step 5 Reflection:
#
# I asked about bankruptcy because the documents do not contain
# financial information. The retrieved documents were not relevant
# to the question. This shows that RAG can still give an answer
# even when the information is missing, so users should verify
# important answers.

# Step 6: Reflection


print("\nStep 6 ***********************************************************")

# Final Reflection
#
# 1. Framework comparison:
#
# The manual semantic RAG lesson required many lines of code for
# chunking, creating embeddings, indexing, and retrieving documents.
# In this project, LlamaIndex handled most of those steps with only
# a few lines of code. This shows the value of a framework because
# it reduces the amount of code needed and makes it easier to build
# and maintain a RAG application.
#
# 2. Another useful business use case:
#
# A useful example would be an employee support assistant for a
# company. It could use internal HR policies, benefits documents,
# employee handbooks, and company procedures to answer employee
# questions quickly without employees having to search through
# many documents.
#
# 3. RAG failure mode:
#
# One failure mode RAG cannot completely prevent is hallucination.
# Even when retrieval is working correctly, the language model can
# misunderstand the retrieved information, combine information
# incorrectly, or give an answer that is not fully supported by
# the documents. This means retrieved information should still be
# checked, especially for important decisions.
# Mini-Project - Groundwork Coffee Co. Q&A Assistant ---

from dotenv import load_dotenv
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


# --- Step 1: Setup ---

print("Step 1 ***********************************************************")

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

docs_dir = Path(
    "../lessons/06_AI_augmentation/resources/groundwork_docs"
)

assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
assert docs_dir.is_dir(), f"Document path is not a directory: {docs_dir}"

Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)


# --- Step 2: Load the Documents ---

print("\nStep 2 ***********************************************************")

documents = SimpleDirectoryReader(
    str(docs_dir)
).load_data()

print(f"Loaded {len(documents)} documents:")

for doc in documents:
    print(f"- {doc.metadata.get('file_name', 'Unknown')}")


# --- Step 3: Build the Index and Query Engine ---

print("\nStep 3 ***********************************************************")

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(
    similarity_top_k=3
)

print("Index built successfully. Ready to answer questions.")


# --- Step 4: Query the Assistant ---

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

    if response.source_nodes:
        node = response.source_nodes[0]

        print(
            f"Document: "
            f"{node.metadata.get('file_name', 'Unknown')}"
        )
        print(f"Similarity Score: {node.score}")
        print(f"Text Preview (200 chars): {node.text[:200]}")
    else:
        print("No source nodes were retrieved.")

# Step 4 Reflection:
# The assistant's answers were mostly accurate and sounded confident.
# One surprising result was that the weekend-hours question retrieved
# our_story.txt as the top node even though the answer came from the
# hours information. This shows that a good-looking answer does not
# always mean the top retrieved chunk was the best match.


# --- Step 5: Find a Failure ---

print("\nStep 5 ***********************************************************")

failure_query = "Is Groundwork facing bankruptcy?"

failure_response = query_engine.query(failure_query)

print("\nQUESTION:")
print(failure_query)

print("\nFULL RESPONSE:")
print(failure_response)

print("\nALL THREE RETRIEVED SOURCE NODES:")

for i, node in enumerate(failure_response.source_nodes[:3], start=1):
    print(f"\nNode {i}")
    print(
        f"Document: "
        f"{node.metadata.get('file_name', 'Unknown')}"
    )
    print(f"Similarity Score: {node.score}")
    print(f"Text Preview (200 chars): {node.text[:200]}")

# Step 5 Reflection:
# I asked about bankruptcy because the Groundwork documents do not
# contain financial information, so I expected retrieval to struggle.
# The retrieved chunks were unrelated to bankruptcy, but the model still
# answered that there was no information instead of inventing financial facts.
#
# The model's tone stayed confident even though the retrieved information
# was not relevant. This shows why users should not trust a confident AI
# answer just because it sounds certain.
#
# To improve the system, I would add a similarity threshold and an
# "information not found" rule so the assistant can refuse to answer
# when the retrieved evidence is too weak.


# --- Step 6: Reflection ---

print("\nStep 6 ***********************************************************")

# 1. Framework comparison:
# The manual RAG pipeline required many lines for chunking, embeddings,
# indexing, and retrieval. In this project, the main LlamaIndex index and
# query-engine setup took only about 5 lines of code, not counting imports,
# setup, and printing. This shows that a framework can greatly reduce
# boilerplate and make a RAG application easier to build and maintain.
#
# 2. Another use case:
# A company could build an employee support assistant using HR policies,
# benefits documents, and employee procedures. Employees could ask
# questions without searching through many internal documents.
#
# 3. RAG failure mode:
# RAG cannot completely prevent hallucination. Even when the correct
# information is retrieved, the LLM can misunderstand it, combine facts
# incorrectly, or give an answer that is not fully supported. Important
# answers should still be checked.
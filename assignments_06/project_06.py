# =============================================================================
# Part 2: Mini-Project - Groundwork Coffee Co. Q&A Assistant
# =============================================================================

from dotenv import load_dotenv
import os
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


# =============================================================================
# Step 1: Setup
# =============================================================================

print("\nStep 1 ***********************************************************")

# Load the API key from the .env file.
load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("API key loaded successfully.")
else:
    print("Warning: OPENAI_API_KEY not found.")


# Path to the Groundwork Coffee documents.
docs_dir = Path(
    "lessons/06_AI_augmentation/resources/groundwork_docs"
)

# Make sure the document directory exists before using it.
assert docs_dir.exists(), (
    f"Document directory not found: {docs_dir}"
)

assert docs_dir.is_dir(), (
    f"Document path is not a directory: {docs_dir}"
)


# Configure the language model and embedding model.
Settings.llm = OpenAI(
    model="gpt-4o-mini"
)

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)


# =============================================================================
# Step 2: Load the Documents
# =============================================================================

print("\nStep 2 ***********************************************************")

documents = SimpleDirectoryReader(
    str(docs_dir)
).load_data()

print(f"\nLoaded {len(documents)} documents:")

# Print the filename of every loaded document.
for document in documents:
    print(
        f"- {document.metadata.get('file_name', 'Unknown')}"
    )


# =============================================================================
# Step 3: Build the Index and Query Engine
# =============================================================================

print("\nStep 3 ***********************************************************")

index = VectorStoreIndex.from_documents(
    documents
)

query_engine = index.as_query_engine(
    similarity_top_k=3
)

print(
    "Index built successfully. "
    "Ready to answer questions."
)


# =============================================================================
# Step 4: Query the Assistant
# =============================================================================

print("\nStep 4 ***********************************************************")

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]


for question in questions:

    print("\n" + "=" * 70)
    print("QUESTION:")
    print(question)

    response = query_engine.query(question)

    print("\nANSWER:")
    print(response)

    print("\nTOP RETRIEVED SOURCE NODE:")

    if response.source_nodes:

        # The first node is the highest-ranked retrieved source.
        node = response.source_nodes[0]

        print(
            f"Document: "
            f"{node.metadata.get('file_name', 'Unknown')}"
        )

        print(
            f"Similarity Score: {node.score}"
        )

        print(
            f"Text Preview: {node.text[:200]}"
        )

    else:
        print("No source nodes were retrieved.")


# Step 4 Reflection:
#
# The assistant gave mostly accurate and confident answers.
# The strongest retrieved documents usually matched the questions well.
# One interesting result was that retrieval does not always put the
# best document first, so the retrieved source should still be checked.


# =============================================================================
# Step 5: Find a Failure
# =============================================================================

print("\nStep 5 ***********************************************************")

# This question is intentionally difficult because the Groundwork
# documents do not contain financial information about bankruptcy.
failure_query = "Is Groundwork facing bankruptcy?"

failure_response = query_engine.query(
    failure_query
)

print("\nQUESTION:")
print(failure_query)

print("\nFULL RESPONSE:")
print(failure_response)

print("\nALL THREE RETRIEVED SOURCE NODES:")

# Print all three retrieved nodes.
for i, node in enumerate(
    failure_response.source_nodes[:3],
    start=1
):

    print(f"\nNode {i}")

    print(
        f"Document: "
        f"{node.metadata.get('file_name', 'Unknown')}"
    )

    print(
        f"Similarity Score: {node.score}"
    )

    print(
        f"Text Preview: {node.text[:200]}"
    )


# Step 5 Reflection:
#
# I asked about bankruptcy because the Groundwork documents do not
# contain financial information. Retrieval returned unrelated documents
# such as the company story, catering information, and menu.
#
# The model did not have evidence to answer the question and said that
# there was no information about bankruptcy. Its answer sounded fairly
# confident, which shows why users should not automatically trust an AI
# response just because it sounds reasonable.
#
# To improve the system, I would add a rule that tells the model to say
# "I don't have enough information" when the retrieved documents do not
# support an answer. I could also add a minimum similarity-score
# threshold before allowing the model to answer.


# =============================================================================
# Step 6: Reflection
# =============================================================================

print("\nStep 6 ***********************************************************")


# Step 6 Reflection:
#
# 1. Framework comparison
#
# The manual semantic RAG lesson required many lines of code for
# chunking, embeddings, indexing, and retrieval. This LlamaIndex
# implementation uses roughly 80-100 lines for the main RAG workflow,
# while LlamaIndex handles much of the chunking, embedding, indexing,
# and retrieval for us. This shows that a framework can save time and
# make a RAG application easier to build and maintain.
#
#
# 2. Another business use case
#
# A useful example would be an employee support assistant for a company.
# It could use HR policies, benefits documents, employee handbooks,
# and company procedures to answer employee questions quickly.
#
#
# 3. RAG failure mode
#
# One failure mode RAG cannot completely prevent is hallucination.
# Even when the correct information is retrieved, the language model
# can misunderstand the context, combine information incorrectly, or
# give an answer that is not fully supported by the documents.
# Important answers should therefore still be checked.


# =============================================================================
# End of Project
# =============================================================================

print("\n" + "=" * 70)
print("Groundwork Coffee Co. Q&A Assistant completed.")
print("=" * 70)
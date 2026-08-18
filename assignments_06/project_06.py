# Assignment 06 - Part 2
# Mini-Project: Groundwork Coffee Co. Q&A Assistant


# ===========================================================================
# Step 1: Setup
# ===========================================================================

from dotenv import load_dotenv
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


print("\nStep 1 ***********************************************************")


# Load the API key from .env
load_dotenv()

print("API key loaded successfully.")


# Groundwork document directory
#
# This relative path assumes the script is run from the Python 200
# project directory.
docs_dir = Path("../../python-200/lessons/06_AI_augmentation/resources/groundwork_docs")

# Verify that the document directory exists before using it.
assert docs_dir.exists(), (
    f"Document directory not found: {docs_dir}"
)

assert docs_dir.is_dir(), (
    f"Document path is not a directory: {docs_dir}"
)


# Configure the LLM and embedding model.
Settings.llm = OpenAI(
    model="gpt-4o-mini"
)

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)


# ===========================================================================
# Step 2: Load the Documents
# ===========================================================================

print("\nStep 2 ***********************************************************")


documents = SimpleDirectoryReader(
    str(docs_dir)
).load_data()


print(f"\nLoaded {len(documents)} documents:")


# Print the name of every loaded document.
for document in documents:
    print(
        f"- {document.metadata.get('file_name', 'Unknown')}"
    )


# ===========================================================================
# Step 3: Build the Index and Query Engine
# ===========================================================================

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


# ===========================================================================
# Step 4: Query the Assistant
# ===========================================================================

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

    response = query_engine.query(
        question
    )

    print("\nANSWER:")
    print(response)

    print("\nTOP RETRIEVED SOURCE NODE:")

    if response.source_nodes:

        node = response.source_nodes[0]

        print(
            "Document: "
            f"{node.metadata.get('file_name', 'Unknown')}"
        )

        print(
            f"Similarity Score: {node.score}"
        )

        print(
            "Text Preview: "
            f"{node.text[:200]}"
        )

    else:
        print("No source nodes were retrieved.")


# Step 4 Reflection:
#
# The assistant gave mostly accurate answers and sounded confident.
# The answers matched the Groundwork documents. One surprising result
# was that the weekend-hours question had our_story.txt as the top node,
# even though faq.txt contains the hours. This shows that the top retrieved
# node is not always the best source.


# ===========================================================================
# Step 5: Find a Failure
# ===========================================================================

print("\nStep 5 ***********************************************************")


failure_query = (
    "Is Groundwork Coffee Co. facing bankruptcy?"
)


failure_response = query_engine.query(
    failure_query
)


print("\nQUESTION:")
print(failure_query)


print("\nFULL RESPONSE:")
print(failure_response)


print("\nALL THREE RETRIEVED SOURCE NODES:")


if failure_response.source_nodes:

    for i, node in enumerate(
        failure_response.source_nodes[:3],
        start=1
    ):

        print(f"\nNode {i}")

        print(
            "Document: "
            f"{node.metadata.get('file_name', 'Unknown')}"
        )

        print(
            f"Similarity Score: {node.score}"
        )

        print(
            "Text Preview: "
            f"{node.text[:200]}"
        )

else:
    print("No source nodes were retrieved.")


# Step 5 Reflection:
#
# I asked about bankruptcy because the Groundwork documents do not contain
# financial information, so I expected the system to struggle.
#
# Retrieval failed to find a source about bankruptcy. The retrieved chunks
# were from unrelated documents such as the story, wholesale/catering,
# and menu documents.
#
# The model still sounded confident and answered that there was no
# information showing that Groundwork was facing bankruptcy. This is
# important because a confident tone does not mean the answer is correct.
#
# I would improve the system by adding a confidence or similarity threshold
# and telling the model to say "I don't have enough information" when the
# retrieved documents are not relevant. I would also make the system cite
# its sources so the user can verify important answers.


# ===========================================================================
# Step 6: Reflection
# ===========================================================================

print("\nStep 6 ***********************************************************")


# Step 6 Reflection
#
# 1. Framework comparison:
#
# In the manual semantic RAG lesson, chunking, embedding, indexing, and
# retrieval required many lines of code. In this project, the main
# LlamaIndex implementation took about 10-15 lines for loading the
# documents, creating the index, and creating the query engine.
# This shows that a framework can greatly reduce the amount of code needed
# and make a RAG application easier to build and maintain.
#
# 2. Another useful use case:
#
# A company could build an employee support assistant using HR policies,
# benefits documents, and employee handbooks. Employees could ask questions
# without searching through many internal documents themselves.
#
# 3. RAG failure mode:
#
# RAG cannot completely prevent hallucination. Even when the correct
# information is retrieved, the language model can misunderstand it,
# combine information incorrectly, or give an answer that is not fully
# supported by the documents. This is why important answers should still
# be verified.
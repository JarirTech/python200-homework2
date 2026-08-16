# ============================================================
# Part 2: Mini-Project
# Groundwork Coffee Co. Q&A Assistant
# ============================================================

from dotenv import load_dotenv
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


# ============================================================
# Step 1: Setup
# ============================================================

print("\nStep 1 ***********************************************************")

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


docs_dir = Path(
    r"C:\Users\bjari\OneDrive\Desktop\python-200"
    r"\lessons\06_AI_augmentation\resources\groundwork_docs"
)

assert docs_dir.exists(), (
    f"Document directory not found: {docs_dir}"
)

assert docs_dir.is_dir(), (
    f"Document path is not a directory: {docs_dir}"
)


Settings.llm = OpenAI(
    model="gpt-4o-mini"
)

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)


# ============================================================
# Step 2: Load the Documents
# ============================================================

print("\nStep 2 ***********************************************************")

documents = SimpleDirectoryReader(
    input_dir=str(docs_dir)
).load_data()

print(f"\nLoaded {len(documents)} documents:")

for doc in documents:
    print(
        f"- {doc.metadata.get('file_name', 'Unknown')}"
    )


# ============================================================
# Step 3: Build the Index and Query Engine
# ============================================================

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


# **********************************************************************
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

    if response.source_nodes:

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


# *****************************************************************************
# Step 4 Reflection


# The assistant gave mostly accurate and confident answers.
# The answers matched the Groundwork documents, but sometimes
# the top retrieved document was not the best document for the
# question. This shows that retrieval can sometimes be imperfect
# even when the final answer is correct.


# ============================================================
# Step 5: Find a Failure
# ============================================================

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

if failure_response.source_nodes:

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

else:
    print("No source nodes were retrieved.")


# ============================================================
# Step 5 Reflection
# ============================================================

# I asked about bankruptcy because the Groundwork documents
# do not contain financial information, so I expected the system
# to struggle with this question.
#
# The retrieval returned unrelated documents, such as the story,
# wholesale information, and menu. This means retrieval found
# text that was somewhat similar but did not contain the answer.
#
# The model still sounded confident and answered that there was
# no information showing that Groundwork was facing bankruptcy.
# This is important because a confident answer can sound correct
# even when the retrieved documents do not support it.
#
# To improve the system, I would add a retrieval confidence
# threshold. If the similarity scores are too low, the system
# should say that it cannot find enough information instead of
# generating an answer.


# ============================================================
# Step 6: Reflection
# ============================================================

print("\nStep 6 ***********************************************************")


# 1. Framework comparison
#
# The manual semantic RAG lesson required many lines of code for
# chunking, embeddings, indexing, and retrieval. LlamaIndex lets
# me do most of those steps with only a few lines. This shows that
# a framework can make RAG applications faster and easier to build
# and maintain.


# 2. Another business use case
#
# One useful use case would be an employee support assistant.
# It could search company policies, benefits information, and
# employee procedures and answer questions without employees
# having to search through many documents themselves.


# 3. RAG failure mode
#
# RAG cannot completely prevent hallucinations. Even when the
# correct information is retrieved, the language model can still
# misunderstand it, combine information incorrectly, or give an
# unsupported answer. Important answers should still be checked.
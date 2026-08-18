
# Mini-Project - Groundwork Coffee Co. Q&A Assistant


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


print("\nStep 1 ***********************************************************")

load_dotenv()

print("API key loaded successfully.")


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


# =================================================================================
# Step 2: Load the Documents


print("\nStep 2 ***********************************************************")

documents = SimpleDirectoryReader(
    str(docs_dir)
).load_data()

print(f"\nLoaded {len(documents)} documents:")

for doc in documents:
    print(
        f"- {doc.metadata.get('file_name', 'Unknown')}"
    )


# =============================================================================
# Step 3: Build the Index and Query Engine


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
            f"Similarity Score: "
            f"{node.score}"
        )

        print(
            f"Text Preview: "
            f"{node.text[:200]}"
        )

    else:
        print("No source nodes were retrieved.")


# Step 4 Reflection:
#
# The assistant gave mostly accurate answers.
# Some top retrieved documents were not the best match, but the model
# still answered correctly. This shows that retrieval can sometimes
# be imperfect even when the final answer is useful.


# =============================================================================
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
    failure_response.source_nodes[:3],
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
# I asked about bankruptcy because the Groundwork documents do not
# contain financial information, so I expected retrieval to struggle.
#
# The system retrieved unrelated documents such as the company story,
# wholesale information, and menu. This means retrieval did not find
# information that could answer the question.
#
# The model still sounded confident and answered that there was no
# information about bankruptcy. This shows that a confident AI answer
# should not automatically be trusted when the retrieved documents
# are not relevant.
#
# I would improve the system by adding a relevance threshold or a
# "not enough information" rule so the assistant can refuse to answer
# when the retrieved documents are not relevant enough.


# =================================================================================
# Step 6: Reflection


print("\nStep 6 ***********************************************************")


# Final Reflection:
#
# 1. Framework comparison:
#
# The manual semantic RAG pipeline required many lines of code for
# chunking, creating embeddings, indexing, and retrieving information.
# In this project, the main LlamaIndex RAG pipeline took about 10 lines
# for loading the documents, building the index, and creating the
# query engine. This shows the value of a framework because it can
# greatly reduce the amount of code and make a RAG application easier
# to build and maintain.
#
# 2. Another useful use case:
#
# A useful example would be an employee support assistant for a company.
# It could answer questions from HR policies, benefits documents,
# employee handbooks, and company procedures.
#
# 3. RAG failure mode:
#
# RAG cannot completely prevent hallucination. Even when the correct
# information is retrieved, the language model can misunderstand it,
# combine information incorrectly, or give an unsupported answer.
# Important answers should therefore still be checked.
# Assignment 06 - Warmup Exercises

from dotenv import load_dotenv
import os
import string
from pathlib import Path


# .env setup

load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("API key loaded successfully.")
else:
    print("Warning: OPENAI_API_KEY not found.")



from pypdf import PdfReader

from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    Settings,
    VectorStoreIndex,
)

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)



Settings.llm = OpenAI(model="gpt-4o-mini")

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)


print("\nPart 1: Warmup Exercises")


# ====================================================================================
# CONCEPTS


# --- Concepts Question 1 ---

print("\n" + "=" * 70)
print("Concepts Question 1")
print("=" * 70)

# Scenario A:
# Best approach: RAG.
#
# A legal team has hundreds of internal policy PDFs that are updated
# regularly, so RAG is a good choice because the assistant can retrieve
# current information from the document collection.

# Scenario B:
# Best approach: Fine-tuning.
#
# The startup has 3,000 examples of its own writing style, so fine-tuning
# can help the model consistently reproduce the company's specific tone.

# Scenario C:
# Best approach: Prompt engineering.
#
# The analyst only needs to ask questions about one short two-page report,
# so a carefully written prompt with the report included should be enough.


# --- Concepts Question 2 ---

print("\n" + "=" * 70)
print("Concepts Question 2")
print("=" * 70)

# A confidently wrong answer is more harmful because a user may trust it
# and act on incorrect information instead of checking the answer.
#
# For example, if an AI confidently gives incorrect financial information
# about an investment, a person could make a decision that causes financial
# loss.


# --- Concepts Question 3 ---

print("\n" + "=" * 70)
print("Concepts Question 3")



#
# Original steps:
# steps = [
#     "Generate a response from the LLM",
#     "Extract text from source documents",
#     "Receive the user's query",
#     "Retrieve the most relevant chunks",
#     "Convert text chunks into embeddings",
#     "Inject retrieved chunks into the prompt",
#     "Split text into chunks",
#     "Embed the user's query",
# ]
#
# Correct order:
# 1. Extract text from source documents
#    Get the text from the PDFs or other source documents.
#
# 2. Split text into chunks
#    Break the documents into smaller pieces that can be retrieved.
#
# 3. Convert text chunks into embeddings
#    Turn each chunk into a vector that represents its meaning.
#
# 4. Receive the user's query
#    The system receives the question from the user.
#
# 5. Embed the user's query
#    Convert the question into an embedding so it can be compared
#    with the document embeddings.
#
# 6. Retrieve the most relevant chunks
#    Find the document chunks that are most similar to the query.
#
# 7. Inject retrieved chunks into the prompt
#    Add the relevant chunks to the prompt given to the LLM.
#
# 8. Generate a response from the LLM
#    The LLM uses the retrieved information to generate the answer.

# ===========================================================================
# KEYWORD RAG

print("\n" + "=" * 70)
print("Keyword RAG")



documents = {
    "menu.txt": (
        "We serve espresso, lattes, cappuccinos, and cold brew. "
        "Pastries include croissants and muffins baked fresh daily. "
        "Oat milk and almond milk are available."
    ),
    "hours.txt": (
        "We are open Monday through Friday from 7am to 7pm. "
        "On weekends we open at 8am and close at 5pm. "
        "We are closed on Thanksgiving and Christmas Day."
    ),
    "hiring.txt": (
        "We are currently hiring baristas and shift supervisors. "
        "Send your resume to jobs@groundworkcoffee.com."
    ),
    "loyalty.txt": (
        "Join our loyalty program to earn one point per dollar spent. "
        "Redeem 100 points for a free drink of your choice."
    ),
}


def simple_keyword_retrieval(query, documents, verbose=True):
    """Retrieve the document with the most overlapping keywords."""

    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to",
        "is", "are", "was", "were", "by", "with", "at", "from", "that",
        "this", "as", "be", "it", "its", "their", "they", "we", "you",
        "your", "our"
    }

    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        word.translate(translator)
        for word in query.lower().split()
        if word not in stopwords
    }

    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []

    for name, content in documents.items():

        content_words = {
            word.translate(translator)
            for word in content.lower().split()
            if word not in stopwords
        }

        overlap = query_words & content_words
        score = len(overlap)

        scores.append((score, name, content))

        if verbose:
            print(
                f"[{name}] overlap={score} -> "
                f"{sorted(overlap)}"
            )

    scores.sort(reverse=True)

    best = next(
        (
            (name, content)
            for score, name, content in scores
            if score > 0
        ),
        None,
    )

    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]

    if verbose:
        print("\nNo overlapping keywords found.")

    return [("None found", "No relevant content.")]


# ---------------------------------------------------------------------------===========
# Keyword Question 1


print("\nKeyword Question 1")
print("-" * 70)

query_1 = "What are your hours on weekends?"

result_1 = simple_keyword_retrieval(
    query_1,
    documents,
    verbose=True
)

print("\nRetrieved Document:")
print(result_1[0][0])

# Reflection:
# The selected document was hours.txt.
# Keyword RAG got this question right because "weekends" appears in
# both the question and hours.txt.


# ---------------------------------------------------------------------------
# Keyword Question 2


print("\nKeyword Question 2")
print("-" * 70)

query_2 = "Do you have anything without caffeine?"

result_2 = simple_keyword_retrieval(
    query_2,
    documents,
    verbose=True
)

print("\nRetrieved Document:")
print(result_2[0][0])

# Reflection:

# No document was selected because none of the query words matched
# the document words after stopwords and punctuation were removed.
#
# Keyword RAG did not get the answer because the menu mentions
# drinks like espresso and cold brew, but it does not use the word
# "caffeine." A semantic retrieval method would do better because
# it can understand the meaning of the question instead of only
# looking for exact matching words.

# ---------------------------------------------------------------------------
# Keyword Question 3

print("\nKeyword Question 3")
print("-" * 70)



query_3 = "How do I sign up for rewards?"

result_3 = simple_keyword_retrieval(
    query_3,
    documents,
    verbose=True
)

print("\nRetrieved Document:")
print(result_3[0][0])

# Prediction:
# I expected loyalty.txt to be selected because "rewards" and
# "loyalty program" have similar meanings. However, keyword
# retrieval only looks for exact word overlap, so it may fail
# even when the words are related.
#
# Result:
# No document was selected because none of the query words
# appeared exactly in the documents after filtering.
#
# This shows why semantic retrieval can be better than keyword
# retrieval for questions that use different words with similar meanings.

# ===========================================================================
# SEMANTIC RAG


print("\n" + "=" * 70)
print("Semantic RAG")
print("=" * 70)


# ---------------------------------------------------------------------------
# Semantic Question 1


print("\nSemantic Question 1")
print("-" * 70)

# 1. What is a vector embedding?
#
# A vector embedding converts text into numbers that represent its meaning.
# Text with similar meanings usually has similar vectors.

# 2. Which chunk is more relevant: 0.85 or 0.30?
#
# The chunk with a score of 0.85 is more relevant because it is more
# similar to the query.

# 3. Why can semantic search find relevant text without exact word matches?
#
# Semantic search compares meaning instead of only comparing exact words.
# For example, "car" and "automobile" have different words but similar
# meanings.


# ---------------------------------------------------------------------------
# Semantic Question 2
# ---------------------------------------------------------------------------

print("\nSemantic Question 2")
print("-" * 70)

# | Feature             | Keyword RAG                         | Semantic RAG                  |
# |---------------------|-------------------------------------|-------------------------------|
# | What is compared?   | Exact word overlap                  | Meaning of the text           |
# | What is retrieved?  | Full document                       | Relevant text chunk           |
# | Synonyms?            | Usually cannot handle them well     | Can handle similar meanings   |
# | Storage format?     | Plain text                          | Numeric vectors               |
# | Relevance score?    | Number of matching keywords         | Similarity score              |


# =============================================================================================
# LLAMAINDEX


print("\n" + "=" * 70)
print("LlamaIndex Warmup")
print("=" * 70)



PDF_DIR = Path("../../python-200/lessons/06_AI_augmentation/resources/brightleaf_pdfs")


assert PDF_DIR.exists(), f"PDF directory not found: {PDF_DIR}"
assert PDF_DIR.is_dir(), f"PDF path is not a directory: {PDF_DIR}"


# ---------------------------------------------------------------------------
# LlamaIndex Question 1


print("\n" + "=" * 70)
print("LlamaIndex Question 1")
print("=" * 70)


pdf_files = sorted(PDF_DIR.glob("*.pdf"))

assert pdf_files, f"No PDF files found in {PDF_DIR}"


def extract_text_from_pdf(path):
    """Extract text from a PDF."""
    reader = PdfReader(str(path))

    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages)


brightleaf_documents = []

for pdf_file in pdf_files:
    text = extract_text_from_pdf(pdf_file)

    brightleaf_documents.append(
        Document(
            text=text,
            metadata={"file_name": pdf_file.name}
        )
    )


print(f"Found {len(brightleaf_documents)} PDF documents.")


brightleaf_index = VectorStoreIndex.from_documents(
    brightleaf_documents
)

brightleaf_query_engine = brightleaf_index.as_query_engine(
    similarity_top_k=3
)


brightleaf_questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]


for question in brightleaf_questions:

    print("\n" + "-" * 70)
    print("QUESTION:")
    print(question)

    response = brightleaf_query_engine.query(question)

    print("\nANSWER:")
    print(response)

    print("\nRetrieved Source Nodes:")

    for i, node in enumerate(response.source_nodes, start=1):
        print(f"\n--- Source Node {i} ---")
        print(f"Document: {node.metadata.get('file_name', 'Unknown')}")
        print(f"Similarity Score: {node.score}")
        print(f"Text Preview: {node.text[:150]}")

    


# Reflection:
#
# Query 1 returned employee_benefits.pdf as the strongest source.
# Query 2 returned security_policy.pdf as the strongest source.


# ---------------------------------------------------------------------------
# LlamaIndex Question 2


print("\n" + "=" * 70)
print("LlamaIndex Question 2")
print("=" * 70)


benefits_query = "What employee benefits does BrightLeaf offer?"


# similarity_top_k = 1
print("\n--- similarity_top_k = 1 ---")

engine_k1 = brightleaf_index.as_query_engine(
    similarity_top_k=1
)

response_k1 = engine_k1.query(benefits_query)

print("\nQUESTION:")
print(benefits_query)

print("\nANSWER:")
print(response_k1)

print("\nSOURCE NODE SCORES:")

for node in response_k1.source_nodes:
    print(
        f"{node.metadata.get('file_name', 'Unknown')}: "
        f"{node.score}"
    )


# similarity_top_k = 5
print("\n--- similarity_top_k = 5 ---")

engine_k5 = brightleaf_index.as_query_engine(
    similarity_top_k=5
)

response_k5 = engine_k5.query(benefits_query)

print("\nQUESTION:")
print(benefits_query)

print("\nANSWER:")
print(response_k5)

print("\nSOURCE NODE SCORES:")

for node in response_k5.source_nodes:
    print(
        f"{node.metadata.get('file_name', 'Unknown')}: "
        f"{node.score}"
    )


# Reflection:
#
# With top_k=1, the answer was already detailed because the most relevant
# benefits document contained the needed information.
#
# With top_k=5, the answer was still good, but additional retrieved
# documents were less relevant.



# ---------------------------------------------------------------------------
# LlamaIndex Question 3
# 

print("\n" + "=" * 70)
print("LlamaIndex Question 3")
print("=" * 70)


q3 = "What is the plan for BrightLeaf to sponsor Manchester United?"

engine_q3 = brightleaf_index.as_query_engine(
    similarity_top_k=3
)

response_q3 = engine_q3.query(q3)

print("\nQUESTION:")
print(q3)

print("\nANSWER:")
print(response_q3)

print("\nTOP 3 SOURCE NODES:")

for i, node in enumerate(
    response_q3.source_nodes[:3],
    start=1
):
    print(f"\nNode {i}")
    print(
        f"Document: "
        f"{node.metadata.get('file_name', 'Unknown')}"
    )
    print(f"Similarity Score: {node.score}")
    print(f"Text Preview: {node.text[:150]}")


# Reflection:
#
# The documents did not contain information about sponsoring Manchester
# United. The retrieved documents were only loosely related, so the system
# correctly said that there was no information about the requested plan.
# This is an example of a question that cannot be answered from the corpus.


# ---------------------------------------------------------------------------

# --- LlamaIndex Question 4 ---
print("LlamaIndex Question 4")
print("=" * 60)

from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

# Create Judge LLM
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluators
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)


# Query 1: Question that should be answered by the documents
q1 = "What employee benefits does BrightLeaf offer?"
response1 = brightleaf_query_engine.query(q1)

faithfulness_result1 = faithfulness_evaluator.evaluate_response(
    query=q1,
    response=response1
)

relevancy_result1 = relevancy_evaluator.evaluate_response(
    query=q1,
    response=response1
)

print("\nQUERY 1")
print(q1)

print("\nFaithfulness Score:")
print(faithfulness_result1.score)

print("\nRelevancy Score:")
print(relevancy_result1.score)


# Query 2: Question that is not in the documents
q2 = "What is BrightLeaf's favorite sports team?"
response2 = brightleaf_query_engine.query(q2)

faithfulness_result2 = faithfulness_evaluator.evaluate_response(
    query=q2,
    response=response2
)

relevancy_result2 = relevancy_evaluator.evaluate_response(
    query=q2,
    response=response2
)

print("\n" + "-" * 60)

print("\nQUERY 2")
print(q2)

print("\nFaithfulness Score:")
print(faithfulness_result2.score)

print("\nRelevancy Score:")
print(relevancy_result2.score)

# Reflection:
#
# Query 1 received good scores because the answer was supported by the BrightLeaf documents.
# Query 2 had a lower relevancy score because the documents do not contain information
# about BrightLeaf's favorite sports team.
# This shows that RAG evaluation checks both whether the answer is supported by the
# retrieved information and whether it actually answers the question.
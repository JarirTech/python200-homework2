# =============================================================================
# Assignment 06 - Warmup Exercises
# =============================================================================

from dotenv import load_dotenv
import os
import string
from pathlib import Path

# Required .env setup
load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("API key loaded successfully.")
else:
    print("Warning: OPENAI_API_KEY not found.")


# =============================================================================
# Imports for LlamaIndex
# =============================================================================

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    Document,
)

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


# Configure LlamaIndex
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)


print("\n" + "=" * 70)
print("PART 1: WARMUP EXERCISES")
print("=" * 70)


# =============================================================================
# --- RAG Concepts ---
# =============================================================================

print("\n" + "=" * 70)
print("CONCEPTS QUESTION 1")
print("=" * 70)

# Scenario A:
#
# Best approach: RAG
#
# A legal team should use RAG because the assistant needs to answer
# questions from a large internal document collection. The documents
# are updated regularly, so RAG can retrieve the current information
# without retraining the model every time the policies change.

# Scenario B:
#
# Best approach: Fine-tuning
#
# Fine-tuning is a good choice because the startup has 3,000 examples
# written in its specific brand voice. The examples can help the model
# learn the company's unusual writing style and produce it more consistently.

# Scenario C:
#
# Best approach: Prompt engineering
#
# Prompt engineering is enough because the analyst only needs to ask
# questions about one short two-page report. There is no need to build
# a larger retrieval system for a single small document.


print("\nConcepts Question 2")
print("-" * 70)

# A confidently wrong answer is more harmful because the user may believe
# the answer is correct and act on it. If the AI says "I am not sure,"
# the user knows that the answer needs to be checked.
#
# For example, if someone asks an AI for financial advice and the AI
# confidently recommends buying a stock based on incorrect information,
# the person could lose money by trusting the answer.


print("\nConcepts Question 3")
print("-" * 70)

# The complete RAG process has these eight steps:
#
# 1. Load documents
#    The system loads the documents that contain information the AI may need.
#
# 2. Split documents into chunks
#    Large documents are divided into smaller pieces that are easier to retrieve.
#
# 3. Create embeddings
#    Each document chunk is converted into a numerical vector representing
#    its meaning.
#
# 4. Store/index the embeddings
#    The embeddings are stored in an index so similar information can be
#    found efficiently.
#
# 5. Receive the user's query
#    The system receives the question or request from the user.
#
# 6. Embed the user's query
#    The user's question is converted into an embedding using the same
#    embedding model used for the document chunks.
#
# 7. Retrieve relevant chunks and inject them into the prompt
#    The query embedding is compared with the stored embeddings, the most
#    relevant chunks are retrieved, and those chunks are added to the prompt.
#
# 8. Generate an answer
#    The language model uses the retrieved context to generate an answer
#    based on the available documents.


# =============================================================================
# --- Keyword RAG ---
# =============================================================================

print("\n" + "=" * 70)
print("KEYWORD RAG")
print("=" * 70)


def simple_keyword_retrieval(query, documents, verbose=True):
    """
    Retrieve a document using simple keyword overlap.
    """

    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to",
        "is", "are", "was", "were", "by", "with", "at", "from", "that",
        "this", "as", "be", "it", "its", "their", "they", "we", "you",
        "your", "our"
    }

    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }

    if verbose:
        print(f"\nQuery: {query}")
        print(f"Query tokens (filtered): {sorted(query_words)}")

    scores = []

    for name, content in documents.items():

        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
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


# Small document collection used for the keyword retrieval exercises.
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


# -----------------------------------------------------------------------------
# Keyword Question 1
# -----------------------------------------------------------------------------

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
#
# The selected document is hours.txt.
# Keyword RAG worked because the query contains "weekends," which also
# appears in hours.txt. The document contains the information needed
# to answer the question.


# -----------------------------------------------------------------------------
# Keyword Question 2
# -----------------------------------------------------------------------------

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
#
# Keyword RAG did not select a useful document because there were no
# overlapping keywords. The menu contains drinks, but it does not use
# the exact word "caffeine." Semantic retrieval would probably work
# better because it compares the meaning of the query with the meaning
# of the document instead of only looking for matching words.


# -----------------------------------------------------------------------------
# Keyword Question 3
# -----------------------------------------------------------------------------

print("\nKeyword Question 3")
print("-" * 70)

# Prediction BEFORE running the retrieval:
#
# Prediction: I expect loyalty.txt to be the correct document because
# the question is about signing up for rewards. However, I predict
# keyword retrieval may fail because loyalty.txt does not contain
# the exact words "sign up."

query_3 = "How do I sign up for rewards?"

result_3 = simple_keyword_retrieval(
    query_3,
    documents,
    verbose=True
)

print("\nRetrieved Document:")
print(result_3[0][0])

# Reflection:
#
# My prediction was correct. The correct document is loyalty.txt,
# but keyword retrieval did not select it because the wording
# "sign up for rewards" does not overlap with the wording in loyalty.txt.
# This shows how keyword retrieval can miss a relevant document when
# different words express the same idea.


# =============================================================================
# --- Semantic RAG ---
# =============================================================================

print("\n" + "=" * 70)
print("SEMANTIC RAG")
print("=" * 70)


# -----------------------------------------------------------------------------
# Semantic Question 1
# -----------------------------------------------------------------------------

print("\nSemantic Question 1")
print("-" * 70)

# 1. What is a vector embedding?
#
# A vector embedding converts text into a list of numbers that represents
# its meaning. Texts with similar meanings usually have embeddings that
# are closer together in vector space.

# 2. Two text chunks have cosine similarity scores of 0.85 and 0.30.
#
# The chunk with a score of 0.85 is more relevant because it has a
# stronger similarity to the query than the chunk with a score of 0.30.

# 3. Why can semantic search find a relevant chunk even when none of
# the exact words from the query appear?
#
# Semantic search compares meaning rather than only exact words.
# For example, "car" and "automobile" are different words but have
# similar meanings, so their embeddings can be close together.


# -----------------------------------------------------------------------------
# Semantic Question 2
# -----------------------------------------------------------------------------

print("\nSemantic Question 2")
print("-" * 70)

# | Feature             | Keyword RAG                         | Semantic RAG                    |
# |---------------------|-------------------------------------|---------------------------------|
# | What is compared?   | Exact word overlap                  | Meaning of the text             |
# | What is retrieved?  | Full document                       | Relevant text chunk             |
# | Can it handle synonyms? | Usually no                      | Yes                             |
# | Storage format      | Plain text dictionary               | Numeric embeddings/index        |
# | Relevance score     | Number of overlapping keywords     | Similarity score                |
#
# Keyword RAG is simple and can work well when the query uses the same
# words as the document. Semantic RAG is more flexible because it can
# find related information even when different words are used.


# =============================================================================
# --- LlamaIndex Warmup ---
# =============================================================================

print("\n" + "=" * 70)
print("LLAMAINDEX WARMUP")
print("=" * 70)


# -----------------------------------------------------------------------------
# LlamaIndex Question 1
# -----------------------------------------------------------------------------

print("\nLlamaIndex Question 1")
print("-" * 70)

from pypdf import PdfReader


def extract_text_from_pdf(path):
    """Extract all text from a PDF."""
    reader = PdfReader(str(path))

    parts = []

    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)

    return "\n".join(parts)


PDF_DIR = Path(
    r"C:\Users\bjari\OneDrive\Desktop\python-200"
    r"\lessons\06_AI_augmentation\resources\brightleaf_pdfs"
)

assert PDF_DIR.exists(), f"{PDF_DIR} not found."
assert PDF_DIR.is_dir(), f"{PDF_DIR} is not a directory."

pdf_files = sorted(PDF_DIR.glob("*.pdf"))

assert pdf_files, f"No PDF files found in {PDF_DIR}."

print(f"Found {len(pdf_files)} PDF files.")

documents_data = []

for pdf in pdf_files:

    text = extract_text_from_pdf(pdf)

    documents_data.append(
        Document(
            text=text,
            metadata={"file_name": pdf.name}
        )
    )

print(f"Loaded {len(documents_data)} documents.")

# Build the BrightLeaf index.
brightleaf_index = VectorStoreIndex.from_documents(
    documents_data
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

    print("\nRETRIEVED SOURCE NODES:")

    # Print exactly the top 3 retrieved source nodes.
    for i, node in enumerate(
        response.source_nodes[:3],
        start=1
    ):

        print(f"\nNode {i}")

        print(
            f"Document: "
            f"{node.metadata.get('file_name', 'Unknown')}"
        )

        print(f"Similarity Score: {node.score}")

        print(
            f"Text Preview: "
            f"{node.text[:150]}"
        )


# Reflections:
#
# Query 1:
# The employee_benefits.pdf chunk was the strongest match because it
# directly contained information about employee benefits. The answer
# sounded confident and detailed, and the retrieved source supported it.
#
# Query 2:
# The security_policy.pdf chunk was the strongest match because it
# directly discussed BrightLeaf's security policies. Some lower-ranked
# chunks were less relevant, showing that retrieval is not always perfect.
#
# Overall, the answers sounded confident and were supported by the
# strongest retrieved chunks.


# -----------------------------------------------------------------------------
# LlamaIndex Question 2
# -----------------------------------------------------------------------------

print("\n" + "=" * 70)
print("LlamaIndex Question 2")
print("=" * 70)

q2 = "What employee benefits does BrightLeaf offer?"


# -------------------------
# similarity_top_k = 1
# -------------------------

print("\n--- similarity_top_k = 1 ---")

query_engine_k1 = brightleaf_index.as_query_engine(
    similarity_top_k=1
)

response_k1 = query_engine_k1.query(q2)

print("\nQUESTION:")
print(q2)

print("\nANSWER:")
print(response_k1)

print("\nRETRIEVED SOURCE NODES:")

for i, node in enumerate(
    response_k1.source_nodes[:1],
    start=1
):

    print(f"\nNode {i}")
    print(
        f"Document: "
        f"{node.metadata.get('file_name', 'Unknown')}"
    )
    print(f"Similarity Score: {node.score}")
    print(f"Text Preview: {node.text[:150]}")


# -------------------------
# similarity_top_k = 5
# -------------------------

print("\n--- similarity_top_k = 5 ---")

query_engine_k5 = brightleaf_index.as_query_engine(
    similarity_top_k=5
)

response_k5 = query_engine_k5.query(q2)

print("\nQUESTION:")
print(q2)

print("\nANSWER:")
print(response_k5)

print("\nRETRIEVED SOURCE NODES:")

for i, node in enumerate(
    response_k5.source_nodes[:5],
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
# The top_k=1 answer was already detailed because the first retrieved
# chunk contained most of the important benefits information.
#
# The top_k=5 answer was also detailed, but the additional chunks were
# less relevant than the first one. This shows that more context is
# not always better. More retrieved information can help, but irrelevant
# chunks can also add noise and potentially confuse the model.


# -----------------------------------------------------------------------------
# LlamaIndex Question 3
# -----------------------------------------------------------------------------

print("\n" + "=" * 70)
print("LlamaIndex Question 3")
print("=" * 70)

q3 = "What is the plan for BrightLeaf to sponsor Manchester United?"

query_engine_q3 = brightleaf_index.as_query_engine(
    similarity_top_k=3
)

response_q3 = query_engine_q3.query(q3)

print("\nQUESTION:")
print(q3)

print("\nANSWER:")
print(response_q3)

print("\nRETRIEVED SOURCE NODES:")

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

    print(
        f"Text Preview: "
        f"{node.text[:150]}"
    )


# Reflection:
#
# The documents do not contain information about a Manchester United
# sponsorship. The system retrieved loosely related BrightLeaf documents,
# but none directly answered the question.
#
# The model correctly said that there was no information about the plan.
# This is a good example of why retrieval results and the generated
# answer should be checked instead of assuming that a confident answer
# is automatically correct.


# -----------------------------------------------------------------------------
# LlamaIndex Question 4
# -----------------------------------------------------------------------------

print("\n" + "=" * 70)
print("LlamaIndex Question 4")
print("=" * 70)

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)


judge_llm = OpenAI(model="gpt-4o-mini")

faithfulness_evaluator = FaithfulnessEvaluator(
    llm=judge_llm
)

relevancy_evaluator = RelevancyEvaluator(
    llm=judge_llm
)


# -------------------------
# Query 1 - expected high quality
# -------------------------

q1 = "What employee benefits does BrightLeaf offer?"

response1 = brightleaf_query_engine.query(q1)

faithfulness_result1 = (
    faithfulness_evaluator.evaluate_response(
        response=response1
    )
)

relevancy_result1 = (
    relevancy_evaluator.evaluate_response(
        query=q1,
        response=response1
    )
)


# -------------------------
# Query 2 - expected lower quality
# -------------------------

q4_lower_quality = (
    "What is BrightLeaf's favorite sports team?"
)

response2 = brightleaf_query_engine.query(
    q4_lower_quality
)

faithfulness_result2 = (
    faithfulness_evaluator.evaluate_response(
        response=response2
    )
)

relevancy_result2 = (
    relevancy_evaluator.evaluate_response(
        query=q4_lower_quality,
        response=response2
    )
)


# Print Query 1 results.

print("\nQUERY 1:")
print(q1)

print("\nANSWER:")
print(response1)

print("\nFaithfulness Score:")
print(faithfulness_result1.score)

print("Faithfulness Passing:")
print(faithfulness_result1.passing)

print("\nRelevancy Score:")
print(relevancy_result1.score)

print("Relevancy Passing:")
print(relevancy_result1.passing)


print("\n" + "-" * 70)


# Print Query 2 results.

print("\nQUERY 2:")
print(q4_lower_quality)

print("\nANSWER:")
print(response2)

print("\nFaithfulness Score:")
print(faithfulness_result2.score)

print("Faithfulness Passing:")
print(faithfulness_result2.passing)

print("\nRelevancy Score:")
print(relevancy_result2.score)

print("Relevancy Passing:")
print(relevancy_result2.passing)


# Reflection:
#
# A faithfulness score of 1.0 means the answer is fully supported
# by the retrieved information. A score of 0.0 means the answer is
# not supported by the provided context.
#
# Relevancy measures whether the answer actually addresses the user's
# question. Faithfulness is different because it measures whether the
# answer is supported by the retrieved information.
#
# Query 1 received high scores because the BrightLeaf documents contain
# information about employee benefits and the answer was supported
# by that information.
#
# Query 2 received lower scores because the documents do not contain
# information about BrightLeaf's favorite sports team. The system does
# not have enough information to answer that question reliably.
#
# More generally, the scores can change because one question may have
# strong supporting evidence in the documents while another question
# may ask for information that is missing.
#
# LLM-as-a-judge means using another language model to evaluate an
# AI-generated response. It is useful for RAG because qualities such
# as faithfulness and relevance are difficult to measure with a simple
# correct/incorrect accuracy score.


# =============================================================================
# END OF WARMUP
# =============================================================================

print("\n" + "=" * 70)
print("Warmup 06 completed.")
print("=" * 70)
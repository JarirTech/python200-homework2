# Assignment 06 - Warmup Exercises

from dotenv import load_dotenv
import os
import string
from pathlib import Path

# ---------------------------------------------------------------------------
# .env setup
# ---------------------------------------------------------------------------

load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("API key loaded successfully.")
else:
    print("Warning: OPENAI_API_KEY not found.")


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LlamaIndex setup
# ---------------------------------------------------------------------------

Settings.llm = OpenAI(model="gpt-4o-mini")

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)


print("\nPart 1: Warmup Exercises")


# ===========================================================================
# CONCEPTS
# ===========================================================================

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
print("=" * 70)

# The provided steps:
#
# steps = [
#     "Load documents",
#     "Split documents into chunks",
#     "Create embeddings",
#     "Store/index the embeddings",
#     "Retrieve relevant chunks",
#     "Generate an answer",
#     "Receive the user's query",
#     "Embed the query",
# ]
#
# Correct order:
#
# 1. Load documents
#    The system loads the documents that contain the information it can use.
#
# 2. Split documents into chunks
#    Large documents are divided into smaller pieces that are easier to search.
#
# 3. Create embeddings
#    Each document chunk is converted into a vector representing its meaning.
#
# 4. Store/index the embeddings
#    The embeddings are stored in an index so relevant chunks can be found.
#
# 5. Receive the user's query
#    The system receives the question that the user wants answered.
#
# 6. Embed the query
#    The user's question is converted into an embedding using the same
#    embedding model.
#
# 7. Retrieve relevant chunks
#    The query embedding is compared with the document embeddings to find
#    the most relevant chunks.
#
# 8. Generate an answer
#    The retrieved chunks are injected into the prompt and given to the
#    language model so it can generate an answer based on the documents.


# ===========================================================================
# KEYWORD RAG
# ===========================================================================

print("\n" + "=" * 70)
print("Keyword RAG")
print("=" * 70)


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


# ---------------------------------------------------------------------------
# Keyword Question 1
# ---------------------------------------------------------------------------

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
# ---------------------------------------------------------------------------

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
# The function selected "None found" because none of the documents
# contained overlapping keywords with the question.
# Keyword RAG did not get a useful result.
# Semantic retrieval would be better because it compares meaning,
# so it could connect related ideas even when the exact words differ.


# ---------------------------------------------------------------------------
# Keyword Question 3
# ---------------------------------------------------------------------------

print("\nKeyword Question 3")
print("-" * 70)

# Prediction BEFORE running the retrieval:
#
# Prediction: loyalty.txt.
# I predict loyalty.txt because the question asks about signing up
# for rewards, which is related to the loyalty program.

query_3 = "How do I sign up for rewards?"

result_3 = simple_keyword_retrieval(
    query_3,
    documents,
    verbose=True
)

print("\nRetrieved Document:")
print(result_3[0][0])

# Reflection AFTER running the retrieval:
#
# My prediction was not correct.
# The function returned "None found" because words such as "sign",
# "up", and "rewards" did not overlap with the words in loyalty.txt.
# This shows that keyword retrieval can miss a relevant document when
# the question uses different wording.


# ===========================================================================
# SEMANTIC RAG
# ===========================================================================

print("\n" + "=" * 70)
print("Semantic RAG")
print("=" * 70)


# ---------------------------------------------------------------------------
# Semantic Question 1
# ---------------------------------------------------------------------------

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


# ===========================================================================
# LLAMAINDEX
# ===========================================================================

print("\n" + "=" * 70)
print("LlamaIndex Warmup")
print("=" * 70)


# Use a relative path so the file can run from the Python 200 project folder.
PDF_DIR = Path("../../python-200/lessons/06_AI_augmentation/resources/brightleaf_pdfs")


assert PDF_DIR.exists(), f"PDF directory not found: {PDF_DIR}"
assert PDF_DIR.is_dir(), f"PDF path is not a directory: {PDF_DIR}"


# ---------------------------------------------------------------------------
# LlamaIndex Question 1
# ---------------------------------------------------------------------------

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

    print("\nTOP 3 RETRIEVED SOURCE NODES:")

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
        print(f"Text Preview: {node.text[:150]}")


# Reflection:
#
# Query 1 returned employee_benefits.pdf as the strongest source.
# Query 2 returned security_policy.pdf as the strongest source.
# The answers were detailed and the top retrieved documents were
# relevant to the questions.


# ---------------------------------------------------------------------------
# LlamaIndex Question 2
# ---------------------------------------------------------------------------

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
#
# More retrieved context is not always better. Extra irrelevant context
# can add noise, so the best top_k value depends on the question.


# ---------------------------------------------------------------------------
# LlamaIndex Question 3
# ---------------------------------------------------------------------------

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
# LlamaIndex Question 4
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("LlamaIndex Question 4")
print("=" * 70)


judge_llm = OpenAI(model="gpt-4o-mini")

faithfulness_evaluator = FaithfulnessEvaluator(
    llm=judge_llm
)

relevancy_evaluator = RelevancyEvaluator(
    llm=judge_llm
)


# First query: expected to produce a strong response.
evaluation_query_1 = (
    "What employee benefits does BrightLeaf offer?"
)

evaluation_engine = brightleaf_index.as_query_engine(
    similarity_top_k=3
)

evaluation_response_1 = evaluation_engine.query(
    evaluation_query_1
)


faithfulness_1 = faithfulness_evaluator.evaluate_response(
    response=evaluation_response_1
)

relevancy_1 = relevancy_evaluator.evaluate_response(
    query=evaluation_query_1,
    response=evaluation_response_1
)


print("\nQUERY 1:")
print(evaluation_query_1)

print("\nANSWER 1:")
print(evaluation_response_1)

print("\nQUERY 1 EVALUATION:")
print(f"Faithfulness Score: {faithfulness_1.score}")
print(f"Faithfulness Passing: {faithfulness_1.passing}")
print(f"Relevancy Score: {relevancy_1.score}")
print(f"Relevancy Passing: {relevancy_1.passing}")


# Second query: expected to produce a lower-quality response because
# the information is not in the BrightLeaf documents.
evaluation_query_2 = (
    "What is BrightLeaf's favorite sports team?"
)

evaluation_response_2 = evaluation_engine.query(
    evaluation_query_2
)


faithfulness_2 = faithfulness_evaluator.evaluate_response(
    response=evaluation_response_2
)

relevancy_2 = relevancy_evaluator.evaluate_response(
    query=evaluation_query_2,
    response=evaluation_response_2
)


print("\n" + "-" * 70)

print("\nQUERY 2:")
print(evaluation_query_2)

print("\nANSWER 2:")
print(evaluation_response_2)

print("\nQUERY 2 EVALUATION:")
print(f"Faithfulness Score: {faithfulness_2.score}")
print(f"Faithfulness Passing: {faithfulness_2.passing}")
print(f"Relevancy Score: {relevancy_2.score}")
print(f"Relevancy Passing: {relevancy_2.passing}")


# Reflection:
#
# Query 1 received high faithfulness and relevancy scores because the
# BrightLeaf documents contain information about employee benefits.
#
# Query 2 is lower quality because the documents do not contain information
# about BrightLeaf's favorite sports team. The comparison shows why
# faithfulness and relevancy are useful for evaluating RAG responses.
#
# Faithfulness measures whether the answer is supported by the retrieved
# information. Relevancy measures whether the answer addresses the question.
#
# LLM-as-a-judge uses another language model to evaluate an AI-generated
# answer. This is useful for RAG because qualities such as faithfulness
# and relevance are difficult to measure with only a simple accuracy score.
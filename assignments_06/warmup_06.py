
# assignments_06/warmup_06.py
# Part 1: Warmup Exercises


from dotenv import load_dotenv
import os
import string
from pathlib import Path

from pypdf import PdfReader

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    Document,
)

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)


# **********************************************************************
# Setup
# **********************************************************************

load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("API key loaded successfully.")
else:
    print("Warning: OPENAI_API_KEY not found.")

Settings.llm = OpenAI(model="gpt-4o-mini")

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)

print("\nPart 1: Warmup Exercises")


# **********************************************************************
# Concepts Question 1
# ============================================================

print("\n" + "=" * 70)
print("Concepts Question 1")
print("=" * 70)

# Scenario A:
# Best approach: RAG
#
# A legal team should use RAG because the information is stored
# in internal documents that are updated regularly. RAG allows
# the assistant to retrieve current information from those PDFs
# instead of relying only on information stored in the model.

# Scenario B:
# Best approach: Fine-tuning
#
# Fine-tuning is a good choice because the company has 3,000
# examples of its own writing. Those examples can help the model
# consistently learn the company's specific writing style.

# Scenario C:
# Best approach: Prompt engineering
#
# Prompt engineering is enough because the analyst only needs
# to ask questions about one short two-page report. There is no
# need to build a larger retrieval system for this small task.

#*************************************************************************
# Concepts Question 2
# ============================================================

print("\n" + "=" * 70)
print("Concepts Question 2")
print("=" * 70)

# A confidently wrong answer is more harmful because the user may
# believe it is correct and act on it without checking the information.
# An answer such as "I am not sure" encourages the user to look for
# another source.
#
# For example, if someone asks an AI about an investment and the AI
# confidently gives incorrect information about a stock, the person
# could make a bad financial decision and lose money.


# ============================================================
# Concepts Question 3
# ============================================================

print("\n" + "=" * 70)
print("Concepts Question 3")
print("=" * 70)

# Correct RAG order:
#
# 1. Load documents
#    The documents are loaded so the system has information to search.
#
# 2. Split documents into chunks
#    Large documents are divided into smaller pieces that are easier
#    to search and provide to the language model.
#
# 3. Create embeddings
#    Each text chunk is converted into a vector that represents
#    its meaning.
#
# 4. Store/index the embeddings
#    The embeddings are stored in an index so similar information
#    can be found efficiently.
#
# 5. Retrieve relevant chunks
#    The user's question is compared with the indexed information
#    to find the most relevant chunks.
#
# 6. Generate an answer
#    The retrieved information is given to the language model so
#    it can generate an answer based on the documents.


# ============================================================
# Keyword RAG
# ============================================================

print("\n" + "=" * 70)
print("Keyword RAG")
print("=" * 70)


def simple_keyword_retrieval(query, documents, verbose=True):
    """
    Retrieve the document with the greatest number of
    overlapping keywords.
    """

    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for",
        "to", "is", "are", "was", "were", "by", "with", "at",
        "from", "that", "this", "as", "be", "it", "its",
        "their", "they", "we", "you", "your", "our"
    }

    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }

    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

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


# Sample Groundwork documents

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


# ============================================================
# Keyword Question 1
# ============================================================

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
# The selected document should be hours.txt because it contains
# information about weekend hours. Keyword retrieval works here
# because words from the question overlap with words in hours.txt.


# ============================================================
# Keyword Question 2
# ============================================================

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
# Keyword RAG does not find a useful document because the important
# words in the question do not overlap with the document wording.
# Therefore, no relevant document is selected.
#
# Semantic retrieval would do better because it compares the meaning
# of the query with the meaning of the documents instead of requiring
# exact keyword matches.


# ============================================================
# Keyword Question 3
# ============================================================

print("\nKeyword Question 3")
print("-" * 70)

# Prediction:
#
# I predict that loyalty.txt is the correct document because the
# question is about signing up for rewards. However, I expect
# keyword retrieval may fail because "sign up" and "rewards" may
# not appear in the document using the same wording.

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
# The keyword search may fail because the query uses words that
# do not exactly match the loyalty document. This confirms that
# keyword retrieval can miss relevant information when different
# words are used to express the same idea.


# ============================================================
# Semantic RAG
# ============================================================

print("\n" + "=" * 70)
print("Semantic RAG")
print("=" * 70)


# ============================================================
# Semantic Question 1
# ============================================================

print("\nSemantic Question 1")
print("-" * 70)

# 1. What is a vector embedding?
#
# A vector embedding converts text into a list of numbers that
# represents the meaning of the text. Similar meanings usually
# result in embeddings that are closer together.

# 2. Two text chunks have cosine similarity scores of 0.85 and 0.30.
#
# The chunk with a score of 0.85 is more relevant because it has
# a stronger similarity to the query than the chunk with a score
# of 0.30.

# 3. Why can semantic search find a relevant chunk even when none
# of the exact words from the query appear?
#
# Semantic search compares meaning instead of only comparing exact
# words. For example, "car" and "automobile" use different words
# but have similar meanings.


# ============================================================
# Semantic Question 2
# ============================================================

print("\nSemantic Question 2")
print("-" * 70)

# | Feature              | Keyword RAG                    | Semantic RAG                  |
# |----------------------|--------------------------------|-------------------------------|
# | What is compared?    | Exact word overlap             | Meaning of the text           |
# | What is retrieved?   | Full document                  | Relevant chunk                |
# | Can it handle        | No, not very well              | Yes                           |
# | synonyms?            |                                |                               |
# | Storage format       | Plain text dictionary          | Numeric vectors/embeddings    |
# | Relevance score      | Number of matching keywords    | Similarity score              |


# ============================================================
# LlamaIndex Question 1
# BrightLeaf PDF Pipeline
# ============================================================

print("\n" + "=" * 70)
print("LlamaIndex Question 1")
print("=" * 70)


def extract_text_from_pdf(path):
    """Extract text from a PDF file."""

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


# Build the BrightLeaf index

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

    print("\nRetrieved Source Nodes:")

    if response.source_nodes:

        for i, node in enumerate(
            response.source_nodes,
            start=1
        ):
            print(f"\nNode {i}")
            print(
                f"Document: "
                f"{node.metadata.get('file_name', 'Unknown')}"
            )
            print(f"Score: {node.score}")
            print(f"Text Preview: {node.text[:200]}")

    else:
        print("No source nodes retrieved.")


# Reflection:
#
# Query 1:
# The answer should be based on the employee benefits document.
# The retrieval scores show which chunks were considered most similar.
#
# Query 2:
# The answer should be based on the security policy information.
# The retrieved chunks may not all be equally relevant, which shows
# that retrieval is not always perfect.


#*********************************************************************
# LlamaIndex Question 2
# Compare similarity_top_k = 1 and 5
# ============================================================

print("\n" + "=" * 70)
print("LlamaIndex Question 2")
print("=" * 70)

benefits_query = "What employee benefits does BrightLeaf offer?"


# top_k = 1

print("\n--- similarity_top_k = 1 ---")

query_engine_k1 = brightleaf_index.as_query_engine(
    similarity_top_k=1
)

response_k1 = query_engine_k1.query(
    benefits_query
)

print("\nQUESTION:")
print(benefits_query)

print("\nANSWER:")
print(response_k1)

print("\nRetrieved Source Nodes:")

for i, node in enumerate(
    response_k1.source_nodes,
    start=1
):
    print(f"\nNode {i}")
    print(f"Score: {node.score}")
    print(f"Text Preview: {node.text[:200]}")


# top_k = 5

print("\n--- similarity_top_k = 5 ---")

query_engine_k5 = brightleaf_index.as_query_engine(
    similarity_top_k=5
)

response_k5 = query_engine_k5.query(
    benefits_query
)

print("\nQUESTION:")
print(benefits_query)

print("\nANSWER:")
print(response_k5)

print("\nRetrieved Source Nodes:")

for i, node in enumerate(
    response_k5.source_nodes,
    start=1
):
    print(f"\nNode {i}")
    print(f"Score: {node.score}")
    print(f"Text Preview: {node.text[:200]}")


# Reflection:
#
# With top_k=1, the system uses only the highest-ranked chunk.
# With top_k=5, the system has more information available.
#
# A larger top_k can provide more context, but some lower-ranked
# chunks may be less relevant. The best value depends on the question.


# *************************************************************************
# LlamaIndex Question 3


print("\n" + "=" * 70)
print("LlamaIndex Question 3")
print("=" * 70)

question_3 = (
    "What is the plan for BrightLeaf to sponsor Manchester United?"
)

response_q3 = brightleaf_query_engine.query(
    question_3
)

print("\nQUESTION:")
print(question_3)

print("\nANSWER:")
print(response_q3)

print("\nRetrieved Source Nodes:")

for i, node in enumerate(
    response_q3.source_nodes,
    start=1
):
    print(f"\nNode {i}")
    print(f"Score: {node.score}")
    print(
        f"Document: "
        f"{node.metadata.get('file_name', 'Unknown')}"
    )
    print(f"Text Preview: {node.text[:200]}")


# Reflection:
#
# This question is difficult because the documents do not contain
# information about a plan to sponsor Manchester United.
#
# The system can still retrieve chunks that are somewhat similar,
# but those chunks do not answer the question. This shows that
# retrieval similarity does not always mean the retrieved information
# contains the actual answer.


#*************************************************************************
# LlamaIndex Question 4
# Evaluation


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


# -------------------------
# Query 1
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


print("\nQUERY 1:")
print(q1)

print("\nFaithfulness Score:")
print(faithfulness_result1.score)

print("Faithfulness Passing:")
print(faithfulness_result1.passing)

print("\nRelevancy Score:")
print(relevancy_result1.score)

print("Relevancy Passing:")
print(relevancy_result1.passing)


# -------------------------
# Query 2
# -------------------------

q2 = "What is BrightLeaf's favorite sports team?"

response2 = brightleaf_query_engine.query(q2)

faithfulness_result2 = (
    faithfulness_evaluator.evaluate_response(
        response=response2
    )
)

relevancy_result2 = (
    relevancy_evaluator.evaluate_response(
        query=q2,
        response=response2
    )
)


print("\n" + "-" * 70)

print("\nQUERY 2:")
print(q2)

print("\nFaithfulness Score:")
print(faithfulness_result2.score)

print("Faithfulness Passing:")
print(faithfulness_result2.passing)

print("\nRelevancy Score:")
print(relevancy_result2.score)

print("Relevancy Passing:")
print(relevancy_result2.passing)


# ============================================================
# LlamaIndex Question 4 Reflection
# ============================================================

# A faithfulness score of 1.0 means the answer is fully supported
# by the retrieved information. A score of 0.0 means the answer
# is not supported by the retrieved information.
#
# Relevancy measures whether the answer addresses the user's
# question. Faithfulness measures whether the answer is supported
# by the retrieved information.
#
# Query 1 asks about employee benefits, which are information that
# should be available in the BrightLeaf documents. Therefore, I
# expect this query to have stronger faithfulness and relevancy.
#
# Query 2 asks about BrightLeaf's favorite sports team. If the
# documents do not contain that information, the answer may not
# be supported by the documents and may receive lower scores.
#
# LLM-as-a-judge means using another language model to evaluate
# an AI-generated response. It is useful for RAG because qualities
# such as faithfulness and relevance are difficult to measure with
# only a simple correct/incorrect accuracy score.
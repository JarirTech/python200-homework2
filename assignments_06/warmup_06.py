# --- Setup ---
from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

from pathlib import Path
import string

from pypdf import PdfReader
from llama_index.core import Document, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


# --- RAG Concepts ---
print("\n" + "=" * 70)
print("Part 1: Warmup Exercises")



# --- Concepts Q1 ---
print("\nConcepts Question 1")

print("-" * 70)
# Scenario A: RAG
# RAG is the best choice because the legal team has many internal PDFs
# that are updated often. The assistant can retrieve the newest policy
# information instead of relying only on the model's training data.

# Scenario B: Fine-tuning
# Fine-tuning is the best choice because the startup has 3,000 examples
# of its own writing. The examples can teach the model the company's
# specific dry and minimalist writing style.

# Scenario C: Prompt engineering
# Prompt engineering is enough because the analyst only needs answers
# about one short report. The report can be included in the prompt
# without building a larger retrieval system.


# --- Concepts Q2 ---
print("\nConcepts Question 2")


# A confidently wrong answer is more harmful because a user may believe
# it and act on it without checking. "I am not sure" tells the user to
# look for more information.
#
# For example, if an AI confidently gives incorrect investment advice,
# a person could follow it and lose money.
#
# The confident tone also matters because people often trust answers
# that sound clear and certain, even when the information is wrong.

print("-" * 70)
# --- Concepts Q3 ---
print("\nConcepts Question 3")


# Complete RAG pipeline in the correct order:
#
# 1. Receive the user's query
#    The system first receives the question that the user wants answered.
#
# 2. Extract text from source documents
#    The system reads the useful text from the documents that will be searched.
#
# 3. Split text into chunks
#    Long documents are divided into smaller pieces so relevant sections
#    can be retrieved instead of searching or sending an entire document.
#
# 4. Convert text chunks into embeddings
#    Each chunk is converted into a vector that represents its meaning.
#
# 5. Embed the user's query
#    The user's question is also converted into a vector so it can be
#    compared with the document chunk embeddings.
#
# 6. Retrieve the most relevant chunks
#    The system compares the query embedding with the stored embeddings
#    and selects the chunks that are most similar.
#
# 7. Inject retrieved chunks into the prompt
#    The selected chunks are added to the prompt as context for the LLM.
#
# 8. Generate a response from the LLM
#    The LLM uses the question and retrieved context to generate the answer.
#
# Provided list from the assignment:
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


# --- Keyword RAG ---
print("\n" + "=" * 70)
print("Keyword RAG")



def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to",
        "is", "are", "was", "were", "by", "with", "at", "from", "that",
        "this", "as", "be", "it", "its", "their", "they", "we", "you",
        "our"
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
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)

    best = next(
        ((name, content) for score, name, content in scores if score > 0),
        None
    )

    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]

    if verbose:
        print("\nNo overlapping keywords found.")

    return [("None found", "No relevant content.")]


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

print("-" * 70)
# --- Keyword Q1 ---
print("\nKeyword Question 1")


query = "What are your hours on weekends?"
result = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:")
print(result[0][0])

# The selected document is hours.txt because it contains the word
# "weekends" and the information about weekend hours. Keyword retrieval
# found it because there was a keyword overlap.

print("-" * 70)

# --- Keyword Q2 ---
print("\nKeyword Question 2")


query_2 = "Do you have anything without caffeine?"
result_2 = simple_keyword_retrieval(query_2, documents, verbose=True)

print("\nSelected document:")
print(result_2[0][0])

# Keyword RAG selected "None found" because none of the important
# query words matched the document vocabulary.
# It did not get the intended answer because the menu has drinks,
# but it does not use the exact phrase "without caffeine."
# Semantic retrieval would do better because it compares meaning,
# so it can connect related ideas even when exact words differ.

print("-" * 70)
# --- Keyword Q3 ---
print("\nKeyword Question 3")


# Prediction before running the code:
# I predict loyalty.txt because it contains the loyalty/rewards information.
# However, keyword retrieval may fail because "sign up for rewards" does
# not use the same words as the loyalty document.

query_3 = "How do I sign up for rewards?"
result_3 = simple_keyword_retrieval(query_3, documents, verbose=True)

print("\nSelected document:")
print(result_3[0][0])

# The prediction was not correct. Keyword retrieval selected "None found"
# because the filtered query words did not overlap with loyalty.txt.
# This shows that keyword retrieval can miss a relevant document when
# different words are used to express the same idea.


# --- Semantic RAG ---
print("\n" + "=" * 70)
print("Semantic RAG")



# --- Semantic Q1 ---
print("\nSemantic Question 1")


# 1. A vector embedding converts text into a list of numbers that
#    represents its meaning. Texts with similar meanings tend to have
#    embeddings that are close together.
#
# 2. The 0.85 chunk is more relevant. A higher cosine similarity score
#    means the chunk is more closely related in meaning to the query
#    than the chunk with a score of 0.30.
#
# 3. Semantic search compares meaning instead of only exact words.
#    For example, "car" and "automobile" are different words but have
#    similar meanings, so their embeddings can be close together.
print("-" * 70)

# --- Semantic Q2 ---
print("\nSemantic Question 2")


# | Feature                    | Keyword RAG                       | Semantic RAG                         |
# |----------------------------|-----------------------------------|--------------------------------------|
# | What is compared?          | Exact word overlap                | Meaning represented by embeddings    |
# | What is retrieved?         | Full document                     | Relevant text chunk                  |
# | Can it handle synonyms?    | No                                | Yes                                  |
# | Storage format             | Plain text dictionary             | Numeric vectors/embeddings           |
# | Relevance score            | Number of overlapping keywords    | Similarity score between embeddings  |


# --- LlamaIndex ---
print("\n" + "=" * 70)
print("LlamaIndex")


# Use a relative path so the file can be moved to another copy of the repo.
PDF_DIR = Path("../lessons/06_AI_augmentation/resources/brightleaf_pdfs")

assert PDF_DIR.exists(), f"{PDF_DIR} not found."
assert PDF_DIR.is_dir(), f"{PDF_DIR} is not a directory."


def extract_text_from_pdf(path):
    """Extract text from all pages of one PDF."""
    reader = PdfReader(str(path))
    parts = []

    for page in reader.pages:
        parts.append(page.extract_text() or "")

    return "\n".join(parts)


pdf_files = sorted(PDF_DIR.glob("*.pdf"))
assert pdf_files, f"No PDF files found in {PDF_DIR}."

print(f"\nFound {len(pdf_files)} PDF files.")

brightleaf_documents = []

for pdf in pdf_files:
    text = extract_text_from_pdf(pdf)
    brightleaf_documents.append(
        Document(
            text=text,
            metadata={"file_name": pdf.name}
        )
    )

print(f"Loaded {len(brightleaf_documents)} documents.")

Settings_llm = OpenAI(model="gpt-4o-mini")
Settings_embed = OpenAIEmbedding(model="text-embedding-3-small")


# --- LlamaIndex Q1 ---
print("\n" + "=" * 70)
print("LlamaIndex Question 1")
print("=" * 70)

index = VectorStoreIndex.from_documents(
    brightleaf_documents,
    embed_model=Settings_embed,
)

query_engine = index.as_query_engine(
    llm=Settings_llm,
    similarity_top_k=3,
)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for question in questions:
    print("\n" + "-" * 70)
    print("QUESTION:")
    print(question)

    response = query_engine.query(question)

    print("\nANSWER:")
    print(response)

    print("\nRETRIEVED SOURCE NODES:")

    for i, node in enumerate(response.source_nodes[:3], start=1):
        print(f"\nNode {i}")
        print(f"Document: {node.metadata.get('file_name', 'Unknown')}")
        print(f"Similarity Score: {node.score}")
        print(f"Text Preview (150 chars): {node.text[:150]}")


# Q1 reflection:
# Query 1: The top retrieved chunk was highly relevant because it came
# from employee_benefits.pdf. The answer sounded confident and detailed.
# Some lower-ranked chunks were less relevant.
#
# Query 2: The top retrieved chunk was highly relevant because it came
# from security_policy.pdf. The answer also sounded confident and specific.
# Some lower-ranked chunks were unrelated, which shows that retrieval
# is not always perfect.


# --- LlamaIndex Q2 ---
print("\n" + "=" * 70)
print("LlamaIndex Question 2")


q2 = "What employee benefits does BrightLeaf offer?"

engine_k1 = index.as_query_engine(
    llm=Settings_llm,
    similarity_top_k=1,
)
response_k1 = engine_k1.query(q2)

print("\n--- similarity_top_k = 1 ---")
print("QUESTION:")
print(q2)
print("\nANSWER:")
print(response_k1)
print("\nSOURCE NODE SCORES:")

for node in response_k1.source_nodes:
    print(node.score)


engine_k5 = index.as_query_engine(
    llm=Settings_llm,
    similarity_top_k=5,
)
response_k5 = engine_k5.query(q2)

print("\n--- similarity_top_k = 5 ---")
print("QUESTION:")
print(q2)
print("\nANSWER:")
print(response_k5)
print("\nSOURCE NODE SCORES:")

for node in response_k5.source_nodes:
    print(node.score)

# Reflection:
# The answers were very similar because the top result already contained
# the important benefits information. More retrieved context is not always
# better because extra unrelated chunks can add noise or distract the model.


# --- LlamaIndex Q3 ---
print("\n" + "=" * 70)
print("LlamaIndex Question 3")


q3 = "What is the plan for BrightLeaf to sponsor Manchester United?"

response_q3 = query_engine.query(q3)

print("\nQUESTION:")
print(q3)

print("\nANSWER:")
print(response_q3)

print("\nALL RETRIEVED SOURCE NODES:")

for i, node in enumerate(response_q3.source_nodes[:3], start=1):
    print(f"\nNode {i}")
    print(f"Document: {node.metadata.get('file_name', 'Unknown')}")
    print(f"Similarity Score: {node.score}")
    print(f"Text Preview (150 chars): {node.text[:150]}")

# Reflection:
# I expected this query to be difficult because the BrightLeaf documents
# do not contain information about Manchester United sponsorship.
# The system correctly said there was no information, although retrieval
# still returned loosely related company documents.
# I would improve the system by adding a similarity threshold and a
# clear "information not found" rule before the LLM generates an answer.


# --- LlamaIndex Q4 ---
print("\n" + "=" * 70)
print("LlamaIndex Question 4")
print("=" * 70)

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)

judge_llm = OpenAI(model="gpt-4o-mini")

faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

# First query: expected to produce a strong answer.
q1 = "What employee benefits does BrightLeaf offer?"
response1 = query_engine.query(q1)

faithfulness_result1 = faithfulness_evaluator.evaluate_response(
    response=response1
)
relevancy_result1 = relevancy_evaluator.evaluate_response(
    query=q1,
    response=response1,
)

print("\nQUERY 1:")
print(q1)
print("\nFaithfulness Score:", faithfulness_result1.score)
print("Relevancy Score:", relevancy_result1.score)


# Second query: expected to produce a lower-quality answer because
# the information is not in the BrightLeaf documents.
q4 = "What is BrightLeaf's favorite sports team?"
response4 = query_engine.query(q4)

faithfulness_result4 = faithfulness_evaluator.evaluate_response(
    response=response4
)
relevancy_result4 = relevancy_evaluator.evaluate_response(
    query=q4,
    response=response4,
)

print("\nQUERY 2:")
print(q4)
print("\nFaithfulness Score:", faithfulness_result4.score)
print("Relevancy Score:", relevancy_result4.score)

# Q4 reflection:
# A faithfulness score of 1.0 means the answer is fully supported by
# the retrieved context. A score of 0.0 means the answer is not supported.
#
# Relevancy measures whether the answer actually addresses the question.
# Faithfulness checks support from the retrieved information, while
# relevancy checks whether the response answers the user's question.
#
# Query 1 should have higher scores because the documents contain
# employee benefits information. Query 2 should have lower scores because
# the documents do not say what BrightLeaf's favorite sports team is.
#
# LLM-as-a-judge means using another LLM to evaluate an AI response.
# It is useful for RAG because qualities such as faithfulness and relevance
# are difficult to measure with a simple right/wrong accuracy score.
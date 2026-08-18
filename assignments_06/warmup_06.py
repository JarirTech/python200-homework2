from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("API key loaded successfully.")
else:
    print("Warning: OPENAI_API_KEY not found.")


from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    Document
)

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from pypdf import PdfReader
import string


Settings.llm = OpenAI(model="gpt-4o-mini")

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)



# Part 1: Warmup Exercises


print("\nPart 1: Warmup Exercises")


# =============================================================================
# Concepts Question 1


print("\n" + "=" * 60)
print("Concepts Question 1")


# Scenario A:
# Best approach: RAG
#
# A legal team has hundreds of internal PDFs that are updated regularly.
# RAG is a good choice because the system can retrieve current information
# from the internal documents without retraining the model every time
# the documents change.

# Scenario B:
# Best approach: Fine-tuning
#
# The startup has 3,000 examples of its own writing. Fine-tuning can help
# the model learn and consistently reproduce the company's specific
# writing style and brand voice.

# Scenario C:
# Best approach: Prompt engineering
#
# The analyst only needs to ask questions about one two-page report.
# Prompting the model with the report is simple and does not require
# building a larger retrieval system.


# =============================================================================
# Concepts Question 2


print("\n" + "=" * 60)
print("Concepts Question 2")


# A confidently wrong answer is more harmful because the user may trust
# the answer and act on it without checking the information.
#
# For example, if an AI confidently gives incorrect financial advice about
# buying a stock, the user could follow that advice and lose money.


# =============================================================================
# Concepts Question 3


print("\n" + "=" * 70)
print("Concepts Question 3")


# Correct order for a semantic RAG system:
#
# 1. Load documents
#    The documents are loaded so the system has information to search.
#
# 2. Split documents into chunks
#    Large documents are divided into smaller pieces that are easier to search.
#
# 3. Create embeddings
#    Each text chunk is converted into a vector representing its meaning.
#
# 4. Store/index the embeddings
#    The embeddings are stored in an index so similar information can be found.
#
# 5. Retrieve relevant chunks
#    The question is compared with the embeddings to find relevant information.
#
# 6. Generate an answer
#    The retrieved information is given to the language model to generate
#    an answer based on the documents.


# =============================================================================
# Keyword RAG


print("\n" + "=" * 70)
print("Keyword RAG")



def simple_keyword_retrieval(query, documents, verbose=True):
    """Retrieve the document with the most keyword overlap."""

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


# =============================================================================
# Keyword Question 1


print("\nKeyword Question 1")
print("-" * 70)

query = "What are your hours on weekends?"

result = simple_keyword_retrieval(
    query,
    documents,
    verbose=True
)

print("\nRetrieved Document:")
print(result[0][0])

# Reflection:
# The selected document was hours.txt.
# Keyword RAG got this question right because the query contained
# the word "weekends", which also appears in hours.txt.


# =============================================================================
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
# Keyword RAG did not select a useful document because none of the
# important query words matched the document text.
# Semantic retrieval would do better because it compares meaning,
# so it could recognize that a question about "without caffeine"
# may relate to available drinks even when the exact words differ.


# =============================================================================
# Keyword Question 3


print("\nKeyword Question 3")
print("-" * 70)

# Prediction:
# I predict loyalty.txt should be the correct document because the question
# is about signing up for rewards. However, keyword retrieval may fail because
# loyalty.txt does not use the exact words "sign up".

query_3 = "How do I sign up for rewards?"

result_3 = simple_keyword_retrieval(
    query_3,
    documents,
    verbose=True
)

print("\nRetrieved Document:")
print(result_3[0][0])

# Reflection:
# My prediction was that loyalty.txt was the correct document, but keyword
# retrieval returned no match. The prediction about the correct document
# was right, but keyword retrieval failed because the wording was different.


# =============================================================================
# Semantic RAG


print("\n" + "=" * 70)
print("Semantic RAG")



# =============================================================================
# Semantic Question 1


print("\nSemantic Question 1")


# 1. What is a vector embedding?
#
# A vector embedding converts text into numbers that represent its meaning.
# Texts with similar meanings usually have embeddings that are closer together.

# 2. Which score is more relevant: 0.85 or 0.30?
#
# A score of 0.85 is more relevant because it shows stronger similarity
# to the query than a score of 0.30.

# 3. Why can semantic search find a relevant chunk when exact words do not match?
#
# Semantic search compares meaning instead of only exact words. For example,
# "car" and "automobile" are different words but have similar meanings.


# =============================================================================
# Semantic Question 2


print("\nSemantic Question 2")


# | Feature             | Keyword RAG                    | Semantic RAG              |
# |---------------------|--------------------------------|---------------------------|
# | What is compared?   | Exact word overlap             | Meaning of the text       |
# | What is retrieved?  | Full document                  | Relevant text chunks      |
# | Synonyms?            | Usually not handled well       | Usually handled better    |
# | Storage format?      | Plain text                     | Numeric vectors           |
# | Relevance score?     | Number of matching keywords    | Similarity score          |


# =====================================================================================
# LlamaIndex Question 1


print("\n" + "=" * 70)
print("LlamaIndex Question 1")



def extract_text_from_pdf(path):
    """Extract text from a PDF."""

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


index = VectorStoreIndex.from_documents(documents_data)

query_engine = index.as_query_engine(
    similarity_top_k=3
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


# Reflection for LlamaIndex Question 1:
#
# Query 1 returned the employee_benefits.pdf document as the top result,
# and the answer was detailed and supported by the retrieved information.
#
# Query 2 returned security_policy.pdf as the top result, so retrieval
# worked well for the security question too.
#
# The lower-ranked documents were sometimes less relevant, showing that
# the top result is usually more useful than the lower-ranked results.


# =============================================================================
# LlamaIndex Question 2


print("\n" + "=" * 70)
print("LlamaIndex Question 2")


query = "What employee benefits does BrightLeaf offer?"


# -------------------- similarity_top_k = 1 --------------------

print("\n--- similarity_top_k = 1 ---")

query_engine_k1 = index.as_query_engine(
    similarity_top_k=1
)

response_k1 = query_engine_k1.query(query)

print("\nQUESTION:")
print(query)

print("\nANSWER:")
print(response_k1)

print("\nSOURCE NODE SCORES:")

for node in response_k1.source_nodes:
    print(
        f"{node.metadata.get('file_name', 'Unknown')}: "
        f"{node.score}"
    )


# -------------------- similarity_top_k = 5 --------------------

print("\n--- similarity_top_k = 5 ---")

query_engine_k5 = index.as_query_engine(
    similarity_top_k=5
)

response_k5 = query_engine_k5.query(query)

print("\nQUESTION:")
print(query)

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
# The answers were similar with top_k=1 and top_k=5.
# More retrieved context was not necessarily better because some of the
# additional documents were less relevant to the employee benefits question.
# In this example, the top result already contained most of the information
# needed to answer the question.


# ==================================================================================
# LlamaIndex Question 3


print("\n" + "=" * 70)
print("LlamaIndex Question 3")


query = "What is the plan for BrightLeaf to sponsor Manchester United?"

query_engine_q3 = index.as_query_engine(
    similarity_top_k=3
)

response_q3 = query_engine_q3.query(query)

print("\nQUESTION:")
print(query)

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
# The documents did not contain information about a Manchester United
# sponsorship. The system therefore returned unrelated but somewhat
# similar documents and correctly said there was no information.
# This shows why retrieval results should be checked when information
# is missing.


# =============================================================================
# LlamaIndex Question 4


print("\n" + "=" * 70)
print("LlamaIndex Question 4")



from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator
)


judge_llm = OpenAI(model="gpt-4o-mini")

faithfulness_evaluator = FaithfulnessEvaluator(
    llm=judge_llm
)

relevancy_evaluator = RelevancyEvaluator(
    llm=judge_llm
)


q1 = "What employee benefits does BrightLeaf offer?"

response1 = query_engine.query(q1)

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


q2 = "What is BrightLeaf's favorite sports team?"

response2 = query_engine.query(q2)

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


print("\nQUERY 1:")
print(q1)

print("\nFaithfulness Score:")
print(faithfulness_result1.score)

print("Relevancy Score:")
print(relevancy_result1.score)


print("\n" + "-" * 70)

print("\nQUERY 2:")
print(q2)

print("\nFaithfulness Score:")
print(faithfulness_result2.score)

print("Relevancy Score:")
print(relevancy_result2.score)


# Reflection:
#
# A faithfulness score of 1.0 means the answer is supported by the
# retrieved information. A score of 0.0 means the answer is not supported.
#
# A relevancy score measures how well the answer addresses the question.
# Faithfulness checks support from the documents, while relevancy checks
# whether the answer addresses the user's question.
#
# Query 1 received high scores because the documents contained information
# about employee benefits. Query 2 received low scores because the documents
# did not contain information about BrightLeaf's favorite sports team.
#
# LLM-as-a-judge uses another language model to evaluate an AI answer.
# This is useful because RAG answers can be evaluated for qualities such
# as faithfulness and relevancy instead of only using a simple correct/
# incorrect accuracy measurement.
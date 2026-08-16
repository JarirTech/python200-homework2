from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("API key loaded successfully.")
else:
    print("Warning: OPENAI_API_KEY not found.")


from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)

from llama_index.llms.openai import OpenAI

from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)


# Requirements:
# pip install openai pypdf python-dotenv "llama-index-core==0.14.10" llama-index-embeddings-openai llama-index-llms-openai


#*** Part 1: Warmup Exercises *********************************************************************
print("Part 1: Warmup Exercises")
#Concepts Question 1 **********************************************************

print('Concepts Question 1 ********************************************************')

# Three teams at a software company are each building a different AI project. Add a comment block to your
#  code that identifies the best approach — prompt engineering, fine-tuning, or RAG — for each scenario, 
#  and gives a 1-2 sentence explanation of your reasoning.

###Scenario A: A legal team wants an assistant that can answer questions about their internal policy library — hundreds
#  of PDFs that are updated every quarter.

# Because the pdfs are updated regularely and those documents are stored internally the best approch for them is to 
# use RAG to build AI project.

### Scenario B: A startup wants their model to write product copy in a very specific brand voice — a dry,
#  minimalist style that does not appear much online. They have 3,000 examples their in-house writers
#  produced over the years.

# For scenario B the best approach is to use fine-tuning because the startup has
# 3,000 examples of its own writing style. Fine-tuning can help the model
# consistently reproduce the company's specific tone and style.

### Scenario C: A data analyst needs to ask an LLM questions about a single two-page report she just received. She does 
# not need this to work for any other document.
# As long as in this scenario the data analyse only want to ask an LLM questions about only 2 pages. 
# The prompt engennering will be enough and a good option for him.


#Concepts Question 2 **********************************************************

print('Concepts Question 2 ********************************************************')

# Why is a confidently wrong answer more harmful than one that says "I am not sure"? Give one example of a real 
# situation where a confident hallucination could cause harm.
#
#Because if the answer was simply 'I am not sure' you will keep looking for the answer. However if Ai just gave you 
# a wrong answer you will accept it and may lead to more issues in your work or in your life in general.
# 
#  An example where a confident hallucination could cause harm is asking AI chatbot about an advice to buy a
# stock or share that suggest will go up in the price but however when the user bought it, it went down which could cause financial 
# loss for the user.
# 

# Because of the tone that AI uses make it looks real and  confident and that 's why the user easily will trust it
# and got misleaded. 



#Concepts Question 3 **********************************************************

print('Concepts Question 3 ********************************************************')

# 1. Load documents
#    The documents are loaded so the system has information to search.
#
# 2. Split documents into chunks
#    Large documents are divided into smaller pieces that are easier
#    to search and use.
#
# 3. Create embeddings
#    Each text chunk is converted into a vector that represents its meaning.
#
# 4. Store/index the embeddings
#    The embeddings are stored in an index so similar information
#    can be found quickly.
#
# 5. Retrieve relevant chunks
#    The user's question is compared with the stored embeddings to
#    find the most relevant information.
#
# 6. Generate an answer
#    The retrieved information is given to the language model so it
#    can generate an answer based on the documents.

#Keyword RAG**********************************************************

#Keyword Question 1
print('Keyword RAG Question 1 ********************************************************')

import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "your", "our"
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
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]



documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On  weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}
query = "What are your hours on weekends?"
result = simple_keyword_retrieval(query, documents, verbose=True)

print("\nretrieval Document:")
print(result[0][0])

# The selected document was hours.txt because it contains the
# information about weekend hours. Keyword retrieval worked because
# the query and document use similar words.

#Keyword RAG**********************************************************
#Keyword Question 2
print('Keyword RAG Question 2 ********************************************************')

query_2 = "Do you have anything without caffeine?"
result_2=  simple_keyword_retrieval(query_2, documents, verbose=True)

# Keyword RAG did not find a matching document because the important
# words in the question did not overlap with the document text.
# No useful document was selected. Semantic retrieval would work
# better because it compares the meaning of the question instead
# of only matching exact words.

#Keyword RAG**********************************************************
#Keyword Question 3
print('Keyword RAG Question 3 ********************************************************')

# Prediction:
# I predict loyalty.txt because the question is about signing up for
# rewards, and loyalty.txt contains information about the loyalty program.
# However, I expect keyword retrieval may fail because the document
# may not contain the exact words "sign up".

query_3 = "How do I sign up for rewards?"
result_3 = simple_keyword_retrieval(query_3, documents, verbose=True)

# Keyword retrieval failed because the filtered query words did not
# overlap with the wording used in loyalty.txt. This shows that
# keyword retrieval can miss relevant documents when different words
# express the same idea.

###############################################################################################################################
#Semantic RAG Concepts
#Semantic Question 1
print('#Semantic Question 1**************************************************************************')

# 1. What is a vector embedding?
#
# A vector embedding converts text into a list of numbers that
# represents its meaning. Texts with similar meanings usually have
# embeddings that are closer together in vector space.

# 2. Two text chunks have cosine similarity scores of 0.85 and 0.30.
#
# The chunk with a score of 0.85 is more relevant because it has a
# stronger similarity to the query than the chunk with a score of 0.30.

# 3. Why can semantic search find a relevant chunk even when none of
# the exact words from the query appear?
#
# Semantic search compares meaning rather than only exact words.
# For example, "car" and "automobile" use different words but have
# similar meanings, so their embeddings can be close together.

#Semantic Question 2
print('#Semantic Question 2**************************************************************************')

# | Feature                    | Keyword RAG                       | Semantic RAG      |
# |----------------------------|-----------------------------------|-------------------|
# | What is compared?          | Exact word overlap                | meaning           |
# | What is retrieved?         | Full document                     | chunk             |
# | Can it handle synonyms?    | No                                | yes               |
# | Storage format             | Plain text dictionary             | numeric           |
# | Relevance score            | Number of overlapping keywords    | cosine similarity |

########################################################################################

#LlamaIndex
#LlamaIndex Question 1
print('LlamaIndex Question 1*****************************************************************')

from pathlib import Path
from pypdf import PdfReader
from llama_index.core import Document


def extract_text_from_pdf(path):
    reader = PdfReader(str(path))

    parts = []

    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)

    return "\n".join(parts)


PDF_DIR = Path(
    r"C:\Users\bjari\OneDrive\Desktop\python-200\lessons\06_AI_augmentation\resources\brightleaf_pdfs"
)

assert PDF_DIR.exists(), f"{PDF_DIR} not found."
assert PDF_DIR.is_dir(), f"{PDF_DIR} is not a directory."

pdf_files = sorted(PDF_DIR.glob("*.pdf"))
assert pdf_files, f"No PDF files found in {PDF_DIR}."

print(f"Found {len(pdf_files)} PDF files.")

documents_data = []

for pdf in pdf_files:
    text = extract_text_from_pdf(pdf)

    print(f"\n{pdf.name}")
    print(text[:300])

    documents_data.append(
        Document(
            text=text,
            metadata={"file_name": pdf.name}
        )
    )

print(f"\nLoaded {len(documents_data)} documents.")

index = VectorStoreIndex.from_documents(documents_data)


query_engine = index.as_query_engine(similarity_top_k=3)
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for question in questions:

    
    print("QUESTION:")
    print()
    print(question)

    response = query_engine.query(question)

    print("\nANSWER:")
    print(response)

    print("\nRetrieved Source Nodes:")

    for i, node in enumerate(response.source_nodes, start=1):
        print(f"\nNode {i}")
        print(f"Score: {node.score}")
        print(f"Text Preview: {node.text[:150]}")


#Query 1 :
# The first chunk was very relevant because it directly talked about employee benefits.
# The response sounded confident and detailed.
# Some lower-ranked chunks were less related to the question.

# Query 2 :
# The top chunk matched the security question well.
# The model gave a detailed answer about security rules and protections.
# A few retrieved chunks were not very relevant, which shows retrieval is not always perfect.

#LlamaIndex Question 2
print('LlamaIndex Question 2*****************************************************************')

#k=1



query = "What employee benefits does BrightLeaf offer?"

query_engine_k1 = index.as_query_engine(similarity_top_k=1)

response_k1 = query_engine_k1.query(query)

print("\nQUESTION:")
print(query)

print("\nANSWER:")
print(response_k1)

print("\nRetrieved Source Nodes:")

for i, node in enumerate(response_k1.source_nodes, start=1):
    print(f"\nNode {i}")
    print(f"Score: {node.score}")
    print(f"Text Preview: {node.text[:150]}")

#k=5
print("LlamaIndex Question 2 - top_k = 5")
print("=====================================================================================" )


query = "What employee benefits does BrightLeaf offer?"

query_engine_k5 = index.as_query_engine(similarity_top_k=5)

response_k5 = query_engine_k5.query(query)

print("\nQUESTION:")
print(query)

print("\nANSWER:")
print(response_k5)

print("\nRetrieved Source Nodes:")

for i, node in enumerate(response_k5.source_nodes, start=1):
    print(f"\nNode {i}")
    print(f"Score: {node.score}")
    print(f"Text Preview: {node.text[:150]}")
#With top_k=1, the system retrieved only the most relevant chunk.
# The answer was still detailed because that chunk already contained
# most of the important information about employee benefits.
############################################################################

#LlamaIndex Question 3
print('LlamaIndex Question 3*****************************************************************')



query = "What is the plan for BrightLeaf to sponsor Manchester United?"

query_engine_q3 = index.as_query_engine(similarity_top_k=3)

response_q3 = query_engine_q3.query(query)

print("\nQUESTION:")
print(query)

print("\nANSWER:")
print(response_q3)

print("\nRetrieved Source Nodes:")

for i, node in enumerate(response_q3.source_nodes, start=1):
    print(f"\nNode {i}")
    print(f"Score: {node.score}")
    print(f"Text Preview: {node.text[:150]}")
###
###The system struggled because the documents did not contain information
# about sponsoring Manchester United.

# It still tried to generate an answer using loosely related company
# information.

# This shows that RAG can sometimes produce weak answers when the
# requested information is not in the documents.
###############################################################################################

#LlamaIndex Question 4
print('LlamaIndex Question 4*****************************************************************')

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator
)



judge_llm = OpenAI(model="gpt-4o-mini")

faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

# query
q1 = "What employee benefits does BrightLeaf offer?"

response1 = query_engine.query(q1)

faithfulness_result1 = faithfulness_evaluator.evaluate_response(response=response1)
relevancy_result1 = relevancy_evaluator.evaluate_response(
    query=q1,
    response=response1
)



q2 = "What is BrightLeaf's favorite sports team?"

response2 = query_engine.query(q2)

faithfulness_result2 = faithfulness_evaluator.evaluate_response(response=response2)
relevancy_result2 = relevancy_evaluator.evaluate_response(
    query=q2,
    response=response2
)

print("\nQUERY1")
print(q1)

print("\nFaithfulness Score:")
print(faithfulness_result1.score)

print("Faithfulness Passing:")
print(faithfulness_result1.passing)

print("\nRelevancy Score:")
print(relevancy_result1.score)

print("Relevancy Passing:")
print(relevancy_result1.passing)


print("\n" + "-" * 65)

print("\nQUERY2")
print(q2)

print("\nFaithfulness Score:")
print(faithfulness_result2.score)

print("Faithfulness Passing:")
print(faithfulness_result2.passing)

print("\nRelevancy Score:")
print(relevancy_result2.score)

print("Relevancy Passing:")
print(relevancy_result2.passing)

###

#What does a faithfulness score of 1.0 mean? What would a score of 0.0 indicate?

# A faithfulness score of 1.0 means the answer is fully supported by
# the retrieved information. A score of 0.0 means the answer is not
# supported by the provided context.
#
#What does a relevancy score measure, and how is it different from faithfulness?

# Relevancy measures whether the answer actually addresses the user's question. Faithfulness is about being supported by the documents,
# while relevancy is about answering the question.
#
#Did the scores change between your two queries? If so, why do you think that happened?

# In my results, Query 1 received 1.0 for both faithfulness and relevancy because the documents contained information about BrightLeaf's employee
# benefits, and the answer matched that information.
#
#What is the "LLM-as-a-judge" approach, and why is it used for RAG evaluation instead of a simple accuracy metric?

# Query 2 received 0.0 for both scores because the BrightLeaf documents did not contain information about the company's favorite sports team.
# This shows that the RAG system could not provide a supported or relevant answer when the information was missing from the documents.
#
# LLM-as-a-judge means using another language model to evaluate the quality of an AI-generated answer. It is useful for RAG because answers can be
# judged for qualities such as faithfulness and relevance, which are harder to measure with a simple correct/incorrect accuracy score.
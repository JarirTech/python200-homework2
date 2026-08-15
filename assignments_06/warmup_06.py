from dotenv import load_dotenv
import os
from pathlib import Path

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


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

#steps = [
# "Receive the user's query",: User writes the question in natural language
# "Embed the user's query",: the model emedbeds the questions into  an embeding vector
#"Extract text from source documents",: The module extract text from the pdf source or other documents
#"Split text into chunks",: The large text extracted got split in chunks
#"Convert text chunks into embeddings",: each chunks get converted to embeding vector
#"Retrieve the most relevant chunks",: the module compares the question to the chunks and retreive the most similar one
# "Inject retrieved chunks into the prompt",: the retreived chunk added into the prompt
#"Generate a response from the LLM",: The response is generated and user got the resonse
#     
# ]


#Keyword RAG**********************************************************

#Keyword Question 1
print('Keyword RAG Question 1 ********************************************************')

import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
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
query = "What are your hours on the weekend?"
result = simple_keyword_retrieval(query, documents, verbose=True)

print("\nretrieval Document:")
print(result[0][0])

# The keyword retriever selected loyalty.txt because "your" was the only
# overlapping keyword. This shows a limitation of keyword retrieval:
# common words can produce an incorrect match even when another document
# is more relevant to the question.


#Keyword RAG**********************************************************
#Keyword Question 2
print('Keyword RAG Question 2 ********************************************************')

query_2 = "Do you have anything without caffeine?"
result_2=  simple_keyword_retrieval(query_2, documents, verbose=True)
# No overlapping keywords found.
#Keyword RAG does not fully understand meaning. The document never
#mentions caffeine-free drinks directly, so the retrieval failed.

#Semantic retrieval would work better because it can understand that
#"without caffeine" is related to beverage options even if the exact
#words do not appear in the text.

#Keyword RAG**********************************************************
#Keyword Question 3
print('Keyword RAG Question 3 ********************************************************')

query_3 = "How do I sign up for rewards?"
result_3 = simple_keyword_retrieval(query_3, documents, verbose=True)

# No overlapping keywords found. for filtred words the module coludn't found any match and this is 
# normal I believe for RAG simple keyword
###############################################################################################################################
#Semantic RAG Concepts
#Semantic Question 1
print('#Semantic Question 1**************************************************************************')

#What is a vector embedding? (1-2 sentences):

### A vector embedding converts text into a list of numbers that represents
# the meaning of the text. Texts with similar meanings usually have
# embeddings that are closer together in vector space.

### Two text chunks have cosine similarity scores of 0.85 and 0.30 with a given query. Which chunk is more
#  relevant, and what does that number tell you about the relationship between the texts?

# Cosine similarity measures how similar two vectors are.
# A score of 0.85 indicates a much stronger similarity to the query
# than a score of 0.30, so the 0.85 chunk is the more relevant match.


### Why can semantic search find a relevant chunk even when none of the exact words from the query appear 
# in the chunk?
# because  semantic RAG gives same score for the meaning like car and automobile will get same number
# it doesn't compares text but compares meaning. 

#Semantic RAG Concepts
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

print("\nQUERY1")
print(q1)

print("\nFaithfulness Score:")
print(faithfulness_result1.passing)

print("Relevancy Score:")
print(relevancy_result1.passing)

# query2
q2 = "What is BrightLeaf's favorite sports team?"

response2 = query_engine.query(q2)

faithfulness_result2 = faithfulness_evaluator.evaluate_response(response=response2)
relevancy_result2 = relevancy_evaluator.evaluate_response(
    query=q2,
    response=response2
)

print("-----------------------------------------------------------------")

print("\nQUERY2")
print(q2)

print("\nFaithfulness Score:")
print(faithfulness_result2.passing)

print("Relevancy Score:")
print(relevancy_result2.passing)

###
# #The first query received True for both scores because the
# response matched the documents and answered the question well.

# The second query received True for faithfulness but False
# for relevancy because the documents did not contain information
# about sports teams.

# This shows that a response can stay grounded in the documents
# while still failing to answer the user's question.
#
# LLM-as-a-judge means using another AI model to check how good
# a response is. It is useful for RAG because answers can be
# correct in different ways
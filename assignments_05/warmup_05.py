# #Jarirtech
# Part 1: Warmup 


import json
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI()




# API Q1
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("API Q1 - First Chat Completion")
print("=" * 50)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": (
                "What is one thing that makes Python a good language "
                "for beginners?"
            )
        }
    ]
)

print("\nResponse:")
print(response.choices[0].message.content)

print("\nModel:")
print(response.model)

print("\nTotal tokens used:")
print(response.usage.total_tokens)


# -------------------------------------------------------------------------
# API Q2 - Temperature


print("\n" + "=" * 70)
print("API Q2 - Temperature")
print("=" * 70)

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temperature in temperatures:

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature
    )

    print(f"\nTemperature {temperature}:")
    print(response.choices[0].message.content)

# Higher temperatures generally produce more varied and creative outputs.
# Lower temperatures generally produce more predictable and consistent outputs.
# If I needed a consistent and reproducible result, I would choose
# temperature=0 because it reduces randomness.


# ------------------------------------------------------------
# API Q3 - Multiple Completions


print("\n" + "=" * 60)
print("API Q3 - Multiple Completions")
print("=" * 60)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": (
                "Give me a one-sentence fun fact about pandas "
                "(the animal, not the library)."
            )
        }
    ],
    n=3,
    temperature=1.0
)

for number, choice in enumerate(response.choices, start=1):
    print(f"\nCompletion {number}:")
    print(choice.message.content)


# --------------------------------------------------------------------------------------------
# API Q4  max_tokens


print("\n" + "=" * 60)
print("API Q4 - max_tokens")
print("=" * 60)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain how neural networks work."
        }
    ],
    max_tokens=15
)

print("\nResponse with max_tokens=15:")
print(response.choices[0].message.content)

# The response is limited and may stop before the explanation is complete
# because max_tokens=15 limits how many tokens the model can generate.
# In a real application, max_tokens can be useful for controlling response
# length, reducing unnecessary output, and helping manage API usage and cost.


# =================================================================================
# System Messages and Personas




# System Q1 - Different Personalities


print("\n" + "=" * 70)
print("System Q1 - Different Personalities")
print("=" * 70)


# First personality
messages = [
    {
        "role": "system",
        "content": (
            "You are a patient, encouraging Python tutor. "
            "You always explain things simply and end with a word "
            "of encouragement."
        )
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nResponse 1 - Patient Python Tutor:")
print(response.choices[0].message.content)


# Second personality
messages = [
    {
        "role": "system",
        "content": (
            "You are a strict but professional Python code reviewer. "
            "Explain programming concepts directly and focus on "
            "technical accuracy."
        )
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nResponse 2 - Strict Python Code Reviewer:")
print(response.choices[0].message.content)

# The user question stayed the same, but the system message changed.
# The first assistant should be encouraging and simple, while the second
# should be more direct and technical. The system message changes the
# assistant's personality, tone, and style.


# ------------------------------------------------------------
# System Q2 - Conversation History


print("\n" + "=" * 70)
print("System Q2 - Conversation History")
print("=" * 70)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "My name is Jordan and I'm learning Python."
    },
    {
        "role": "assistant",
        "content": (
            "Nice to meet you, Jordan! Python is a great choice. "
            "What would you like to work on?"
        )
    },
    {
        "role": "user",
        "content": "Can you remind me what my name is?"
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nResponse:")
print(response.choices[0].message.content)

# The API is stateless and does not remember Jordan from a previous call.
# The model knows Jordan's name because the full conversation history is
# passed to the API in the messages list. The previous user message
# containing Jordan's name is included in that list, giving the model
# the context needed to answer the question.


# =============================================================================
# Prompt Engineering
# ============================================================


reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]


# ------------------------------------------------------------
# Prompt Q1 - Zero-Shot



reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Classify the sentiment of each review as exactly one of:
positive, negative, or mixed.

Return exactly three lines, one sentiment per review, in the same order.

Reviews:
1. {reviews[0]}
2. {reviews[1]}
3. {reviews[2]}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

results = response.choices[0].message.content.strip().splitlines()

print("\nPROMPT Q1 — ZERO-SHOT RESULTS")

for i, result in enumerate(results, start=1):
    print(f"Review {i}: {result}")

# No examples are provided. The model classifies each review independently.
# The results are printed separately with the review number.

# ------------------------------------------------------------
# Prompt Q2 - One-Shot


print("\n" + "=" * 70)
print("Prompt Q2 - One-Shot Sentiment")
print("=" * 70)

one_shot_prompt = f"""
Classify the sentiment of each review as positive, negative, or mixed.

Use this example to understand the expected format:

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now classify these reviews.

Review 1: {reviews[0]}
Review 2: {reviews[1]}
Review 3: {reviews[2]}

Return each result using this format:
Review 1: [sentiment]
Review 2: [sentiment]
Review 3: [sentiment]
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": one_shot_prompt
        }
    ]
)

one_shot_result = response.choices[0].message.content

print("\nOne-Shot Results:")
print(one_shot_result)

# Adding one example gives the model a clearer pattern for how the answer
# should be formatted. This can make the output more consistent than
# zero-shot prompting.


# ------------------------------------------------------------===================
# Prompt Q3 - Few-Shot


print("\n" + "=" * 70)
print("Prompt Q3 - Few-Shot Sentiment")
print("=" * 70)

few_shot_prompt = f"""
Classify the sentiment of each review as positive, negative, or mixed.

Use these three examples:

Example 1:
Review: "The product was excellent and arrived quickly."
Sentiment: positive

Example 2:
Review: "The product broke immediately and customer support was unhelpful."
Sentiment: negative

Example 3:
Review: "The price was excellent, but shipping took too long."
Sentiment: mixed

Now classify these reviews:

Review 1: {reviews[0]}
Review 2: {reviews[1]}
Review 3: {reviews[2]}

Return each result using this format:
Review 1: [sentiment]
Review 2: [sentiment]
Review 3: [sentiment]
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": few_shot_prompt
        }
    ]
)

few_shot_result = response.choices[0].message.content

print("\nFew-Shot Results:")
print(few_shot_result)

# Zero-shot is useful when the task is simple and examples are unnecessary.
# One-shot is useful when one example can clarify the expected format or task.
# Few-shot is useful when several examples are needed to demonstrate patterns,
# especially when the task has multiple categories or requires consistent output.


# ------------------------------------------------------------

# Prompt Q4 — Chain of Thought

math_prompt = """
Solve the following problem step by step.
After your explanation, write the final result on a separate line beginning
with exactly: FINAL ANSWER:

A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": math_prompt}]
)

full_response = response.choices[0].message.content

print("\nPROMPT Q4 — CHAIN OF THOUGHT")
print("Full Response:")
print(full_response)

# Find the model-generated final answer
final_answer = "Not found"

for line in full_response.splitlines():
    if line.strip().upper().startswith("FINAL ANSWER:"):
        final_answer = line.strip()
        break

print("\nModel-Generated Final Answer:")
print(final_answer)

# Asking the model to work through the problem step by step can help it
# break the calculation into smaller steps and reduce mistakes on
# multi-step problems.


# ------------------------------------------------------------

# Prompt Q5 — Structured Output

import json

review = (
    "I've been using this tool for three months. It handles large datasets well, "
    "but the UI is clunky and the export options are limited."
)

prompt = f"""
Analyze the following review.

Return ONLY valid JSON.
Do not include markdown or any explanation outside the JSON.

The JSON must contain exactly these keys:
- "sentiment"
- "confidence"
- "reason"

"confidence" must be a float from 0 to 1.
"reason" must be exactly one sentence.

Review:
{review}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

raw_response = response.choices[0].message.content

print("\nPROMPT Q5 — STRUCTURED OUTPUT")
print("RAW RESPONSE:")
print(raw_response)

try:
    data = json.loads(raw_response)

    print("\nParsed JSON Fields:")
    print(f"Sentiment: {data['sentiment']}")
    print(f"Confidence: {data['confidence']}")
    print(f"Reason: {data['reason']}")

except (json.JSONDecodeError, KeyError, TypeError) as e:
    print("\nJSON parsing failed.")
    print(f"Error: {e}")
    print("RAW RESPONSE FOR DEBUGGING:")
    print(raw_response)


# The model is instructed to return only valid JSON with the required keys.
# The raw response is printed first so it can be inspected if JSON parsing fails.

# ------------------------------------------------------------
# Prompt Q6 - Delimiters
# ----------------------

print("\n" + "=" * 70)
print("Prompt Q6 - Delimiters")
print("=" * 70)

# First test: instructional text

user_text = (
    "First boil a pot of water. Once boiling, add a handful of salt and the "
    "pasta. Cook for 8-10 minutes until al dente. Drain and toss with your "
    "sauce of choice."
)

delimiter_prompt = f"""
You will be given text inside triple backticks.

If it contains step-by-step instructions, rewrite them as a numbered list.

If it does not contain instructions, respond with exactly:
"No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": delimiter_prompt
        }
    ]
)

print("\nInstructional Text Result:")
print(response.choices[0].message.content)


# Second test: clearly non-instructional prose

non_instruction_text = (
    "The company opened a new office downtown last year. "
    "The building has large windows and provides employees with "
    "a comfortable workspace."
)

non_instruction_prompt = f"""
You will be given text inside triple backticks.

If it contains step-by-step instructions, rewrite them as a numbered list.

If it does not contain instructions, respond with exactly:
"No steps provided."

```{non_instruction_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": non_instruction_prompt
        }
    ]
)

print("\nNon-Instructional Text Result:")
print(response.choices[0].message.content)

# Delimiters clearly separate the user's text from the instructions.
# This helps prevent the model from confusing the content being analyzed
# with the instructions it should follow.


# =======================================================================
# Local Models with Ollama



# ------------------------------------------------------------
# Ollama Q1


print("\n" + "=" * 70)
print("Ollama Q1 - Local Model")
print("=" * 70)

# Run this command in the terminal:
#
# ollama run qwen3:0.6b "Explain what a large language model is in two sentences."
#
# The output below was pasted from the Ollama terminal run.

"""
Ollama Terminal Output:

A large language model is an artificial intelligence system trained on vast
amounts of text data, enabling it to understand and generate human-like
language, perform tasks like writing, answering questions, or even playing
games, and adapt to various contexts.
"""

ollama_prompt = (
    "Explain what a large language model is in two sentences."
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": ollama_prompt
        }
    ]
)

print("\nOpenAI Response:")
print(response.choices[0].message.content)

# The OpenAI response is more detailed and mentions several language-related
# tasks, while the Ollama response is shorter and uses simpler wording.
#
# One advantage of running a model locally is that data can remain on the
# local computer, which can provide greater privacy and reduce dependence
# on an external API.
#
# One disadvantage is that a small local model can be less capable or less
# detailed than larger cloud-based models. Local models may also require
# enough computer memory and processing power to run efficiently.
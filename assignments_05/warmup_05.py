#Jarirtech
###Part 1: Warmup Exercises*******************************************

#The Chat Completions API

###API Question 1******************************************************
print("API Question 1***************************************************************")
from dotenv import load_dotenv
from openai import OpenAI
import json
load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)
print("response: ", response.choices[0].message.content)

#print the name of the model that responded
print("model: ", response.model)



#total number of tokens used
print("number of token: ", response.usage.total_tokens)

##API Question 2***************************************************************
print("API Question 2***************************************************************")

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]
for t in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],  temperature=t
             )
    print(f"\nTemperature = {t}")
    print(response.choices[0].message.content)


# for temperature = 0.7 and 1.5 the output changes every time, however for temperature = 0 the output
#  is stable. If I want a consistent, reproducible output I will chose t = 0.


##API Question 3***************************************************************
print("API Question 3***************************************************************")

#Use n=3 with temperature=1.0 to get three different completions in a single API call. Print all three.
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)
#Iterate over response.choices and print each one.
for i, choice in enumerate(response.choices, start=1):
     print(f"Response {i}:") 
     print(choice.message.content)


##API Question 4****************************************************************************
print("API Question 4*****************************************************************************")
#Set max_tokens=15 and send a prompt that would normally produce a long response (for example, "Explain how neural networks work."). Print the result. 
# Add a comment: What happened, and why might you want to use max_tokens in a real application?

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15)
print(response.choices[0].message.content)

# the response is incomplete and stoped at 15 token.
# i may use token to reduce the cost and have shorter and fast responses
###########################################################################################
#System Messages and Personas

###System Question 1*******************************************************************************
print("System Question 1*******************************************************************************")
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things "
    "simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create( model="gpt-4o-mini", messages=messages )
print("System Question 1 (patient Tutor ): ") 
print(response.choices[0].message.content)

# 
messages = [
    {
        "role": "system",
        "content": (
            "You are a strict, concise Python instructor. "
            "Give direct technical explanations without encouragement."
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

print("\nStrict Instructor Response:")
print(response.choices[0].message.content)

# The response changes because the system message changes
# the personality, tone, and style of the assistant

###System Question 2*******************************************************************************
print("System Question 2*******************************************************************************")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
response = client.chat.completions.create( model="gpt-4o-mini", messages=messages )
print(response.choices[0].message.content)

##Comment:
## the system know Jordan name because the name was included in the messages history as a list 
#*************************************************************************************************
#Prompt Engineering


#Prompt Engineering Question 1.  Zero-Shot*******************************************************************************
print("Prompt Engineering Question 1*******************************************************************************")



reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1):

    prompt = f"""
Classify the sentiment of the review below as exactly one of:
positive, negative, or mixed.

Do not provide an explanation.
Return only the sentiment.

Review:
"{review}"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    print(f"Review {i} Sentiment:")
    print(response.choices[0].message.content)

###Prompt Engineering Question 2.  one-Shot*******************************************************************************
print("Prompt Engineering Question 2*******************************************************************************")

example = """
Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed
"""

for i, review in enumerate(reviews, start=1):

    prompt = f"""
{example}

Classify the following review as positive, negative, or mixed.
Return only the sentiment.

Review: "{review}"
Sentiment:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    print(f"Review {i} Sentiment:")
    print(response.choices[0].message.content)

# The one-shot example helps the model understand the expected
# output format and makes the responses more consistent.


# Prompt Engineering Question 3.Few-Shot  ****************************************************
print("Prompt Engineering Question 3********************************************************")

examples = """
Example 1:
Review: "The support team was friendly and solved my issue quickly."
Sentiment: positive

Example 2:
Review: "The application crashes every day and customer support never replies."
Sentiment: negative

Example 3:
Review: "The price is reasonable, but the interface is difficult to use."
Sentiment: mixed
"""

for i, review in enumerate(reviews, start=1):

    prompt = f"""
{examples}

Classify the following review as positive, negative, or mixed.
Return only the sentiment.

Review: "{review}"
Sentiment:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    print(f"Review {i} Sentiment:")
    print(response.choices[0].message.content)

# Zero-shot is useful when the task is simple and well understood.
# One-shot is useful when one example is enough to demonstrate the format.
# Few-shot is useful when multiple examples are needed to demonstrate
# different categories and improve consistency.


###Prompt Engineering Question 4.Chain of Thought  *******************************************************************************
print("Prompt Engineering Question 4*******************************************************************")

prompt = """
Solve this problem step by step.

A data engineer earns $85,000 per year. She gets a 12% raise,
then 6 months later takes a new job that pays $7,500 more per
year than her post-raise salary.

Clearly label the final answer as:
FINAL ANSWER:
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)

# Breaking the calculation into smaller steps can help the model
# avoid arithmetic mistakes and arrive at a more accurate answer.



###Prompt Engineering Question 5.Few-Shot  *******************************************************************************
print("Prompt Engineering Question 5*******************************************************************")

review = (
    "I've been using this tool for three months. "
    "It handles large datasets well, but the UI is clunky "
    "and the export options are limited."
)

prompt = f"""
Analyze the review below.

Return ONLY valid JSON.
Do not use markdown or code fences.

The JSON must contain exactly these three keys:
- "sentiment": a string containing "positive", "negative", or "mixed"
- "confidence": a float between 0 and 1
- "reason": one sentence explaining the sentiment

Review:
"{review}"
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

raw_response = response.choices[0].message.content

print("\nRaw Response:")
print(raw_response)

try:
    parsed = json.loads(raw_response)

    print("\nParsed JSON Fields:")
    print("Sentiment:", parsed["sentiment"])
    print("Confidence:", parsed["confidence"])
    print("Reason:", parsed["reason"])

except json.JSONDecodeError as e:
    print("\nInvalid JSON response.")
    print("Error:", e)
    print("Raw response:")
    print(raw_response)


###Prompt Engineering Question 6.Delimiters  *******************************************************************************
print("Prompt Engineering Question 6*******************************************************************")

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
response = client.chat.completions.create( 
    model="gpt-4o-mini", 
    messages=[{"role": "user", "content": prompt}] )
print(response.choices[0].message.content)

fake_text = "The team was very competitive  today even they lost the game"
prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{fake_text}```
"""

response = client.chat.completions.create( 
    model="gpt-4o-mini", 
    messages=[{"role": "user", "content": prompt}] )
print(response.choices[0].message.content)
#
# Delimiters clearly separate the user's text from the instructions.
# This helps prevent the model from confusing the user's content
# with the instructions it needs to follow.


#*****************************************************************************************************

#Local Models with Ollama

#Ollama Question 1
print('Ollama Question 1')

""" Ollama Output:
A large language model is an artificial intelligence system trained on vast amounts of text data, enabling it to
understand and generate human-like language, perform tasks like writing, answering questions, or even playing
games, and adapt to various contexts.  """

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain what a large language model is in two sentences."
        }
    ]
)

print("\nOpenAI Response:")
print(response.choices[0].message.content)


# Difference:
# Both responses explain what an LLM is, but the OpenAI response gives
# more detail about how LLMs work and mentions tasks like translation
# and summarization. The Ollama response is shorter and simpler.
# Advantage:
# Running a model locally can provide more privacy because the data
# does not need to be sent to an external API.
# Disadvantage:
# Local models can be less capable or less detailed than larger
#Jarirtech
###Part 1: Warmup Exercises*******************************************

#The Chat Completions API

###API Question 1******************************************************
print("API Question 1***************************************************************")
from dotenv import load_dotenv
from openai import OpenAI

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

messages = [
    {"role": "system", "content": "You are a unpatient, encouraging Python tutor."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create( model="gpt-4o-mini", messages=messages )
print("System Question 1 (unpatient tutor): ") 
print(response.choices[0].message.content)


## comment:
## the output changed when the system personality changed

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


###Prompt Engineering Question 1.  Zero-Shot*******************************************************************************
print("Prompt Engineering Question 1*******************************************************************************")

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]
for i, review in enumerate(reviews, start=1):
    prompt = f"classify the sentiment of each review below as positive, negative, or mixed:  {review}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    print(f"\nReview {i}:")
    print(response.choices[0].message.content)

###Prompt Engineering Question 2.  one-Shot*******************************************************************************
print("Prompt Engineering Question 2*******************************************************************************")

example = """Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed"""

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]
for i, review in enumerate(reviews, start=1):
    prompt = f''' {example}

    Review: "{review}" Sentiment: '''
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

    print(f"\nReview {i}:")
    print(response.choices[0].message.content)

# the use of one exmple help on formating but the ouput was not as expected as for the second review 
# negative but the output was mixed
###Prompt Engineering Question 3.Few-Shot  *******************************************************************************
print("Prompt Engineering Question 3*******************************************************************")

examples = ''' Review: "Player score nice goal." Sentiment: positive

Review: "The player miss the goal." Sentiment: negative

Review: "The player did a nice kick but missed the goal." Sentiment: mixed '''

for i, review in enumerate(reviews, start=1):
    prompt = f''' {examples}

    Review: "{review}" Sentiment: '''
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    print(f"\nReview {i}:")
    print(response.choices[0].message.content)

## using few shots gave consistant output

###Prompt Engineering Question 4.Chain of Thought  *******************************************************************************
print("Prompt Engineering Question 4*******************************************************************")
prompt = ''' Solve this problem step by step and clearly label the final answer.

A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later takes 
a new job that pays $7,500 more per year than her post-raise salary. What is her final annual salary? '''

response = client.chat.completions.create( 
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}] )
print(response.choices[0].message.content)
#Why does asking the model to reason step by step tend to improve accuracy on problems like this?
# breaking down the problem helped the model to solve it with accuraccy
###Prompt Engineering Question 5.Few-Shot  *******************************************************************************
print("Prompt Engineering Question 5*******************************************************************")

import json
#return the result only as valid JSON with keys sentiment, confidence (a float from 0 to 1), 
# and reason (one sentence)

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

prompt = f''' Analyze the review and return ONLY valid JSON.

Required keys:

sentiment
confidence
reason

Review: "{review}" '''
response = client.chat.completions.create( 
    model="gpt-4o-mini", 
    messages=[{"role": "user", "content": prompt}] )

raw_response = response.choices[0].message.content 
print("Raw Response: ", raw_response) 


try: 
    parsed = json.loads(raw_response)

    print("\nSentiment:", parsed["sentiment"])
    print("Confidence:", parsed["confidence"])
    print("Reason:", parsed["reason"])

except json.JSONDecodeError:
    print("\nInvalid JSON response.") 
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
#Using delimeters to separate commands from the user's text stops the model from getting confused. 
# This keeps the model focused on the real rules

#*****************************************************************************************************

#Local Models with Ollama

#Ollama Question 1
print('Ollama Question 1')

""" Example Ollama Output:
A large language model is an AI system trained on massive amounts of text so it can understand and 
generate human language. It can answer questions, write text, summarize information, and help with 
many language tasks. """
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[ { "role": "user", "content": "Explain what a large language model is in two sentences." } ])

print(response.choices[0].message.content)

##OpenAI’s answer felt more like a real person and gave more info.Ollama’s answer was quick and easy to
#  understand.
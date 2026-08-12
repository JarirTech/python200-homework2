# Mini-Project — Job Application Helper (Jarirtech )

# Task 1: Setup and System Prompt

import json
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
print("API key loaded:", bool(os.getenv("OPENAI_API_KEY")))

client = OpenAI()

SYSTEM_PROMPT = """
You are a job application coach helping applicants and people looking for a job.

Your goals:
- Make resume bullet points better
- Help write cover letters
- Improve job application materials
- Answer career questions

How to act:
- Only talk about jobs and careers
- Never make up fake experience or skills
- Be honest and professional
- Remind users to check and edit everything you write
- Remind users that every industry is different and they should use their own judgment
- Be helpful, realistic, and keep your answers short
"""

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )

    return response.choices[0].message.content


# I instructed the assistant to avoid inventing information because
# AI tools can hallucinate accomplishments or skills. This helps keep
# the generated content ethical and realistic.


# **************************************************************************



# Task 2: Bullet Point Rewriter

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    """ Rewrite resume bullets and return a parsed list of dictionaries."""

    # Format bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""

    You are a professional resume coach helping a career changer.

    Rewrite each resume bullet point below to be:

    More specific
    More results-oriented
    More compelling
    Professional sounding

    Use strong action verbs.

    Do not invent facts that aren't implied by the original.
    Do not invent numbers, percentages, dates, achievements, awards, or results.

    Respond ONLY with a valid JSON list.
    Do not include markdown or explanations.
    Do not wrap the list inside another object such as {{"items": [...]}}.

    Each item in the list must contain exactly two keys:

    "original"
    "improved"

    Here are the bullet points:

        {bullet_text}

    """

    messages = [{"role": "user", "content": prompt}]

    # Calling the model
    response = get_completion(messages)

    # Keeping the original response for debugging
    raw_response = response

    print("\nRAW RESPONSE:")
    print(raw_response)

    # Removing code fences if the model adds them
    cleaned_response = (
        raw_response
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        # Parse the JSON into a Python list
        results = json.loads(cleaned_response)

        # Making sure the response is a list
        if not isinstance(results, list):
            raise ValueError("Expected a JSON list.")

        print("\nRewritten Resume Bullets:")

        # Printing each original bullet next to its improved version
        for item in results:
            print(f"\nORIGINAL: {item['original']}")
            print(f"IMPROVED: {item['improved']}")
            print("-" * 50)

        # Returning the parsed list
        return results

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print("\nCould not process the JSON response.")
        print("Error:", e)
        print("\nRaw response:")
        print(raw_response)

        return []
# --- Testing ---

starter_bullets = [
"Helped customers with their problems",
"Made reports for the management team",
"Worked with a team to finish the project on time"
]

print("\n**** Testing ****")

rewritten_bullets = rewrite_bullets(starter_bullets)

print("\nReturned Result:")
print(rewritten_bullets)

# The original bullets are not good because they do not clearly show

# achievements, impact, or measurable results.

# The improved bullets are better because they highlight clarity, professionalism, and

# action-oriented language without inventing unsupported accomplishments.




#********************************************************************************
#Task 3: Cover Letter Generator

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages)


print("\n**** TASK 3 TEST ***")

# testing 
job_title = "Junior Data Engineer"

background = (
    "Five years of experience as a middle school math teacher; "
    "recently completed a Python course and built data pipelines "
    "using Prefect and Pandas."
)

cover_letter = generate_cover_letter(job_title, background)

print("\nGenerated Cover Letter Opening:\n")
print(cover_letter)

# I chose examples that show strong transitions from one field into another.
# The few-shot examples help control tone, structure, and specificity.
# They guide the model away from generic corporate language and toward
# realistic career-change storytelling.

   
#*************************************************************************************

#Task 4: Moderation Check
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    # returning True if safe, False if flagged, and print a message if flagged
    if flagged:
        print("\nYour message was flagged by the moderation system.")
        print("Please rephrase your request respectfully and safely.\n")
        return False

    return True


print("\n***TASK 4 TEST ***")

safe_test = "Can you help improve my resume bullet points?"
unsafe_test = "How can I hurt someone?"

print("Safe input result:", is_safe(safe_test))
print("Unsafe input result:", is_safe(unsafe_test))


#*****************************************************************************

#Task 5: The Chatbot Loop



def run_chatbot():


    # Initialize conversation history
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)

    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. General job application questions")

    print("\nType 'quit' at any time to exit.\n")

    while True:

        user_input = input("You: ").strip()

        # Exit chatbot
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # Ignore empty input
        if not user_input:
            continue

        # Check the user's message with moderation
        if not is_safe(user_input):
            continue

        # -------------------------------------------------
        # Resume bullet feature
        # -------------------------------------------------
        if "bullet" in user_input.lower() or "resume" in user_input.lower():

            print("\nPaste your bullet points below, one per line.")
            print("Type 'DONE' when finished.\n")

            raw_bullets = []

            while True:
                line = input().strip()

                if line.upper() == "DONE":
                    break

                if line:
                    raw_bullets.append(line)

            if raw_bullets:
                rewritten_bullets = rewrite_bullets(raw_bullets)

                print("\nJob Application Helper:")
                print("Your rewritten bullets are shown above.")

            else:
                print("\nNo bullet points were provided.")

        # -------------------------------------------------
        # Cover letter feature
        # -------------------------------------------------
        elif "cover letter" in user_input.lower():

            job_title = input(
                "\nJob Application Helper: What is the job title? "
            ).strip()

            background = input(
                "Job Application Helper: Briefly describe your background: "
            ).strip()

            result = generate_cover_letter(job_title, background)

            print("\nGenerated Cover Letter Opening:")
            print(result)

            print("\nReminder: Review and edit this before submitting it.")

        # -------------------------------------------------
        # Regular conversation
        # -------------------------------------------------
        else:

            # 1. Append the user's message to conversation history
            messages.append({
                "role": "user",
                "content": user_input
            })

            # 2. Send the conversation history to the model
            reply = get_completion(messages)

            # 3. Print the assistant's response
            print(f"\nJob Application Helper: {reply}\n")

            # 4. Append the assistant's response to conversation history
            messages.append({
                "role": "assistant",
                "content": reply
            })


# --- Main Program ---

if __name__ == "__main__":
    run_chatbot()



#******************************************************************************

#Task 6: Ethics Reflection

# --- Task 6: Ethics Reflection ---

# AI job application tools can sometimes give biased advice or favor

# certain writing styles, industries, or types of experience. They can

# also generate incorrect or exaggerated information if users do not

# carefully review the results.

#

# One important guardrail is reminding users to review and edit all

# generated content before submitting it to an employer. Another

# important guardrail is telling the AI not to invent skills,

# accomplishments, numbers, or work experience that the user does not

# actually have.

#

# Users should also use their own judgment because hiring practices

# and expectations can differ between industries and employers.

# AI should help improve a job application, but the final application

# should accurately represent the applicant's real experience and skills.

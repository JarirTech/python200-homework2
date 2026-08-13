# # Mini-Project — Job Application Helper (Jarirtech )


import json
from dotenv import load_dotenv
from openai import OpenAI



# Task 1: Setup and System Prompt


load_dotenv()

client = OpenAI()


SYSTEM_PROMPT = """
You are a professional job application coach helping job seekers,
career changers, and applicants improve their job application materials.

Your responsibilities include:
- Improving resume bullet points
- Helping write cover letter openings
- Answering questions about job applications
- Helping users present their real experience clearly and professionally

Behavioral guidelines:
- Stay focused on jobs, careers, and job application materials.
- Never invent experience, skills, accomplishments, credentials, or results.
- Use only information provided by the user.
- Always remind users to review and edit AI-generated content before
  submitting it to an employer.
- Explain that industry expectations and communication styles can vary,
  so users should use their own judgment.
- Keep responses helpful, realistic, professional, and concise.
"""


def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    """Send a conversation to the OpenAI API and return the text response."""

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )

    return response.choices[0].message.content


# I specifically instructed the assistant not to invent experience,
# skills, accomplishments, or results because AI can generate information
# that sounds realistic but is not true. This guardrail helps keep resume
# and job application materials honest and based on the user's real experience.


# ================================================================================
# Task 2: Bullet Point Rewriter


def rewrite_bullets(bullets: list[str]) -> list[dict]:
    """
    Rewrite resume bullets and return a parsed JSON list.

    Each returned item contains:
    - original
    - improved
    """

    # Format the bullets into a delimited block.
    bullet_text = "\n".join(
        f"- {bullet}" for bullet in bullets
    )

    prompt = f"""
You are a professional resume coach helping a career changer.

Rewrite each resume bullet point below to be:
- More specific
- More results-oriented
- More compelling
- Professional sounding

Use strong action verbs.

Do not invent facts, accomplishments, skills, numbers, or results
that are not supported by the original bullet.

Return ONLY a valid JSON list.
Do not include markdown, explanations, or a preamble.

Each item must contain exactly two keys:
"original"
"improved"

User's bullet points are inside the following delimiters:
{bullet_text}

"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = get_completion(messages)

    try:
        results = json.loads(response)

        # Confirm that the model returned a JSON list.
        if not isinstance(results, list):
            raise ValueError("Expected a JSON list.")

        # Confirm each item has the required keys.
        for item in results:
            if not isinstance(item, dict):
                raise ValueError("Each item must be a dictionary.")

            if "original" not in item or "improved" not in item:
                raise ValueError(
                    "Each item must contain 'original' and 'improved'."
                )

        print("\n" + "=" * 70)
        print("REWRITTEN RESUME BULLETS")
        print("=" * 70)

        for number, item in enumerate(results, start=1):

            print(f"\nBullet {number}")
            print(f"ORIGINAL : {item['original']}")
            print(f"IMPROVED : {item['improved']}")
            print("-" * 70)

        return results

    except (json.JSONDecodeError, ValueError, TypeError, KeyError) as error:

        print("\nJSON parsing failed.")
        print("Error:", error)

        print("\nRAW RESPONSE FOR DEBUGGING:")
        print(response)

        return []


# Test Task 2
starter_bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

print("\n" + "=" * 70)
print("TASK 2 TEST")
print("=" * 70)

rewritten_bullets = rewrite_bullets(starter_bullets)

print("\nReturned Python list:")
print(rewritten_bullets)

# These bullets are weak because they are vague and do not clearly
# communicate impact, results, or specific actions. The model should
# improve them by using stronger action verbs and clearer descriptions.
# It should not invent numbers or accomplishments that were not provided.


# ============================================================
# Task 3: Cover Letter Generator


def generate_cover_letter(job_title: str, background: str) -> str:
    """Generate a cover letter opening paragraph using few-shot prompting."""

    prompt = f"""
You write strong cover letter opening paragraphs for career changers.

The paragraph should be:
- 3-5 sentences
- Confident
- Specific
- Professional
- Free of clichés
- Based only on the person's actual background

Here are two examples of the style and tone you should match:

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.

Opening:
After seven years as a registered nurse, I've spent my career making
decisions under pressure using incomplete information — which turns out
to be excellent training for data analysis. I recently completed a data
analytics program where I built dashboards tracking patient outcomes
across departments. I'm excited to bring that combination of clinical
context and technical skill to [Company]'s mission-driven work.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations, self-taught Python developer for two years.

Opening:
I spent a decade on the operations side of banking, watching technology
decisions get made by people who had never processed a wire transfer or
resolved a failed ACH batch. That frustration turned into curiosity, and
two years of self-teaching Python later, I'm ready to be on the other side
of those decisions. I'm applying to [Company] because your work on payment
infrastructure is exactly where my domain expertise and new technical
skills intersect.

Now write an opening paragraph for this person:

Role: {job_title}

Background:
{background}

Opening:
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    return get_completion(messages)


# Test Task 3
job_title = "Junior Data Engineer"

background = (
    "Five years of experience as a middle school math teacher; "
    "recently completed a Python course and built data pipelines "
    "using Prefect and Pandas."
)

print("\n" + "=" * 70)
print("TASK 3 TEST")
print("=" * 70)

cover_letter = generate_cover_letter(
    job_title,
    background
)

print("\nGenerated Cover Letter Opening:")
print(cover_letter)

# I chose examples that demonstrate realistic career transitions and
# show how previous experience can connect to a new technical role.
# The few-shot examples help control the tone, structure, specificity,
# and level of detail of the generated cover letter.


# ============================================================
# Task 4: Moderation Check


def is_safe(text: str) -> bool:
    """Return True if the text is not flagged by moderation."""

    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    flagged = result.results[0].flagged

    if flagged:
        print(
            "\nYour message was flagged by the moderation system."
        )
        print(
            "Please rephrase your request respectfully and safely.\n"
        )
        return False

    return True


# Test Task 4

safe_test = "Can you help improve my resume bullet points?"
unsafe_test = "How can I hurt someone?"

print("\n" + "=" * 70)
print("TASK 4 TEST")
print("=" * 70)

print("\nSafe input:")
print(safe_test)
print("Result:", is_safe(safe_test))

print("\nFlagged input:")
print(unsafe_test)
print("Result:", is_safe(unsafe_test))


# ==================================================================


# Task 5: The Chatbot Loop
# ===============================================================

def run_chatbot():
    # Initialize conversation history with the system prompt.
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
        # Get user input.
        user_input = input("You: ").strip()

        # ----------------------------------------------------------------
        # Handle exit
        # ----------------------------------------------------
        if user_input.lower() in {"quit", "exit"}:
            print(
                "\nJob Application Helper: "
                "Good luck with your applications!"
            )
            break

        # ----------------------------------------------------==
        # Skip empty input
        # -------------------------------------------------------------
        if not user_input:
            continue

        # -----------------------------------------------------------
        # Moderation check
        # ----------------------------------------------------
        if not is_safe(user_input):
            continue

        # =====================================================================
        # Resume Bullet Point Feature
        # ====================================================
        if "bullet" in user_input.lower() or "resume" in user_input.lower():

            print(
                "\nJob Application Helper: "
                "Paste your bullet points below, one per line."
            )
            print("When you're done, type 'DONE' on its own line.\n")

            raw_bullets = []

            while True:
                line = input().strip()

                if line.upper() == "DONE":
                    break

                if line:
                    raw_bullets.append(line)

            # Make sure the user entered at least one bullet.
            if not raw_bullets:
                print(
                    "\nJob Application Helper: "
                    "No bullet points were provided.\n"
                )
                continue

            # Add the user's bullet-rewrite request to conversation history.
            bullet_request = (
                f"{user_input}\n\n"
                "Bullet points:\n"
                + "\n".join(f"- {bullet}" for bullet in raw_bullets)
            )

            messages.append({
                "role": "user",
                "content": bullet_request
            })

            # Rewrite the bullets using the helper function.
            rewritten_bullets = rewrite_bullets(raw_bullets)

            # Check whether the function returned results.
            if not rewritten_bullets:
                print(
                    "\nJob Application Helper: "
                    "I could not rewrite those bullet points.\n"
                )

                # Record the failed interaction.
                messages.append({
                    "role": "assistant",
                    "content": (
                        "I could not rewrite the provided resume "
                        "bullet points."
                    )
                })

                continue

            # Build a readable assistant response for the conversation history.
            bullet_response_parts = []

            for item in rewritten_bullets:
                original = item["original"]
                improved = item["improved"]

                bullet_response_parts.append(
                    f"ORIGINAL: {original}\n"
                    f"IMPROVED: {improved}"
                )

            bullet_response = "\n\n".join(bullet_response_parts)

            # Print the results.
            print("\nJob Application Helper: Rewritten Resume Bullets\n")
            print(bullet_response)

            # Add the assistant's response to conversation history.
            messages.append({
                "role": "assistant",
                "content": bullet_response
            })

            print()

        # ============================================================
        # Cover Letter Feature
        # ====================================================
        elif "cover letter" in user_input.lower():

            job_title = input(
                "\nJob Application Helper: "
                "What is the job title? "
            ).strip()

            background = input(
                "Job Application Helper: "
                "Briefly describe your background: "
            ).strip()

            # Add the user's cover-letter request to conversation history.
            cover_letter_request = (
                f"{user_input}\n\n"
                f"Job title: {job_title}\n"
                f"Background: {background}"
            )

            messages.append({
                "role": "user",
                "content": cover_letter_request
            })

            # Generate the cover letter opening.
            result = generate_cover_letter(
                job_title,
                background
            )

            # Print the generated result.
            print("\nGenerated Cover Letter Opening:\n")
            print(result)

            # Add the assistant's response to conversation history.
            messages.append({
                "role": "assistant",
                "content": result
            })

            print(
                "\nReminder: Review and edit this before submitting it."
            )
            print()

        # ====================================================
        # Regular Conversation
        # =================================================================
        else:

            # Add the user's message to conversation history.
            messages.append({
                "role": "user",
                "content": user_input
            })

            # Send the complete conversation history to the model.
            reply = get_completion(messages)

            # Print the assistant's response.
            print(
                f"\nJob Application Helper: {reply}\n"
            )

            # Add the assistant's response to conversation history.
            messages.append({
                "role": "assistant",
                "content": reply
            })


# ========================================================================
# Running the chatbot
# ============================================================

if __name__ == "__main__":
    run_chatbot()

# ===============================================================
# Task 6: Ethics Reflection


#
# AI job application tools can produce biased advice because their training data
# may favor certain communication styles, industries, or cultural backgrounds.
# A job seeker could also submit inaccurate or exaggerated information if they
# do not carefully review the generated content before sending it to an employer.
# One important guardrail is to clearly remind users to review and edit every
# response and verify that it accurately represents their real experience and skills.
# I would also keep moderation enabled to reduce the risk of harmful or inappropriate
# requests.
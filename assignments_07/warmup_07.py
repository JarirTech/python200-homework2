# Jarirtech
# Warmup 07


from pathlib import Path
import json
import os

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dotenv import load_dotenv
from openai import OpenAI

from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel, tool

from datetime import datetime
import json

from scipy.stats import pearsonr
from pathlib import Path





# .env setup
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

client = OpenAI()
print('OpenAI client created.')


#Tool Definitions and the ReAct Loop
print("Warmup 07 started")



# Tool Definitions and the ReAct Loop


# ---------Q1----------------------------------------------------------------
# ------------------------------------------------------------
print('---------Q1----------------------------------------------------------------')
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert Celsius to Fahrenheit."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Converts a temperature from Celsius to Fahrenheit.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "Temperature in Celsius"
                }
            },
            "required": ["celsius"]
        }
    }
}

print('Tools list defined with one tool: celsius_to_fahrenheit')



print(f"celsius_to_fahrenheit(0) = {celsius_to_fahrenheit(0)}")
print(f"celsius_to_fahrenheit(100) = {celsius_to_fahrenheit(100)}")
print(f"celsius_to_fahrenheit(-40) = {celsius_to_fahrenheit(-40)}")


# --------------------------------------------------------------------
# Q2: 

print("\nQ2-----")



def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

get_current_time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Returns the current local time as a string.",
        "parameters": {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    }

print('Tools list defined with one tool: get_current_time')
tools = [
    get_current_time_schema,
    
]



def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time.'''
    
    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: get_current_time
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''

# Prediction:
# The model should not call get_current_time because the question
# is about temperature, not the current time.
# I expect one API call because no tool should be needed.




response = run_agent(
    "Convert 100 degrees Celsius to Fahrenheit"
)

print("Result:", response)

### comment: 
# The model was able to answer the question without calling the get_current_time tool, and used only one API call as expected.


# # ------------------------------------------------------------
# # Q3


print('====== Q3===========================================================')


tools = [
    get_current_time_schema,
    celsius_to_fahrenheit_schema
]


def run_agent(user_prompt: str) -> str:

    SYSTEM_PROMPT = '''
    You are a helpful assistant.
    Use tools when needed.
    '''

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # First API call
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',
    )

    first_message = first_response.choices[0].message

    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Check if tool was requested
    if first_message.tool_calls:

        print("Agentic mode engaged...")

        for tool_call in first_message.tool_calls:

            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name == 'get_current_time':
                tool_result = get_current_time()

            elif function_name == 'celsius_to_fahrenheit':
                tool_result = celsius_to_fahrenheit(
                    arguments['celsius']
                )

            else:
                tool_result = f'Error: unknown tool {function_name}.'

            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Second API call
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )

        final_message = second_response.choices[0].message
        return final_message.content or ''

    return first_message.content or ''


response_a = run_agent( "What is 37 degrees Celsius in Fahrenheit?")

print("Response A:", response_a)

# A tool was called because the model needed
# the temperature conversion tool.


response_b = run_agent(
    "What is the boiling point of water in plain English?"
)

print("Response B:", response_b)

# the module gave an answer without calling any tool


# ==================Multi-Tool Agent======================================
# Q4

print('====== Q4===========================================================')





class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.
    
        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error
    
        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."
    
        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"
    
        # If someone accidentally passes x == y, treat it like "plot y"
        if x == y:
            x = None
    
        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."
    
        title_csv = self.csv_name or "current CSV"
    
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."
    
        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"
    
        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()
        
        return f"Plotted {y} vs {x} as a {plot_type}."

    #==========compute_correlation===================================================================
    
    def compute_correlation(self, col1: str, col2: str):

        """
        Compute the Pearson correlation between two columns
        in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """

        error = self._ensure_loaded()
        if error:
            return error

        if col1 not in self.df.columns:
            return {
                "error": f"'{col1}' is not a column. Options: {self.df.columns.tolist()}"
            }

        if col2 not in self.df.columns:
            return {
                "error": f"'{col2}' is not a column. Options: {self.df.columns.tolist()}"
            }

        r, p = pearsonr(self.df[col1], self.df[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(r, 4),
            "p_value": round(p, 4)
        }


print("Class defined")

resources_dir = Path( r"../../python-200\lessons\07_AI_agents\resources")

csv_manager = CsvManager(resources_dir)

print("CsvManager created")

# Tool schemas
tools_schema = [

    {
        'type': 'function',
        'function': {
            'name': 'load_csv',
            'description': 'Load a CSV file.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'filename': {
                        'type': 'string'
                    }
                },
                'required': ['filename']
            }
        }
    },

    {
        'type': 'function',
        'function': {
            'name': 'compute_correlation',
            'description': 'Compute correlation between two columns.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'col1': {'type': 'string'},
                    'col2': {'type': 'string'}
                },
                'required': ['col1', 'col2']
            }
        }
    }
]

# Node tools
node_tools = {
    'load_csv': csv_manager.load_csv,
    'compute_correlation': csv_manager.compute_correlation
}

print("compute_correlation tool added successfully.")


# Q5
# ============================================================
print('====== Q5===========================================================')

def run_agent_cycle(messages, user_text, max_tool_rounds=5):

    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):

        content = json.dumps(result, default=str) if not isinstance(result, str) else result

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

        return tool_message

    for loop_idx in range(max_tool_rounds):

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        assistant_entry = {
            "role": "assistant",
            "content": msg.content
        }

        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                tc.model_dump() for tc in msg.tool_calls
            ]

        messages.append(assistant_entry)

        # no tool calls = final answer
        if not msg.tool_calls:
            return msg.content

        # ACT + OBSERVE
        for tool_call in msg.tool_calls:

            name = tool_call.function.name

            tool_args = json.loads(
                tool_call.function.arguments or "{}"
            )

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)

            if fn is None:
                result = {"error": f"Tool '{name}' not found."}

            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()

                except Exception as e:
                    result = {
                        "error": f"Tool failed: {e}"
                    }

            messages.append(
                observe_tool_result(tool_call.id, result)
            )

    return "I hit the tool-round limit."

SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)

# Start conversation with system prompt
messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]

# Run the agent
result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh."
)

# Print final answer
print(result)

# Q6
# ============================================================
print('====== Q6===========================================================')

# system:
# gives instructions to the agent

# user:
# the user ask the question

# assistant:
# the model reasoning and responses

# tool:
# output returned from a tool call

import json

print(json.dumps(messages, indent=2, default=str))


# Q7
# ============================================================
print('====== Q7===========================================================')


# smolagents tool


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute Pearson correlation between two columns.

    Args:
        col1 (str): First column name.
        col2 (str): Second column name.

    Returns:
        dict: Correlation result.
    """

    return csv_manager.compute_correlation(col1, col2)


print(compute_correlation.description)

# smolagents created the tool description automatically
# from the function name and docstring.

# In Q4 we wrote the JSON schema manually.

# Good docstrings help smolagents
# create better tool descriptions.


# ============================================================
# Q8


print('====== Q8===========================================================')

# Create model

model = OpenAIServerModel(
    model_id="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Tools list

TOOLS = [
    compute_correlation
]

# ToolCallingAgent

tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model
)

# CodeAgent

code_agent = CodeAgent(
    tools=TOOLS,
    model=model
)

prompt = (
    "Load bike_commute.csv. "
    "Plot avg_heart_rate vs duration_min "
    "as a scatter plot with green dots."
)

response_tool = tool_agent.run(prompt)

response_code = code_agent.run(
    prompt,
    additional_args={"csv_manager": csv_manager}
)

print("ToolCallingAgent Response:")
print(response_tool)

print("\nCodeAgent Response:")
print(response_code)


# The ToolCallingAgent did not create the plot or change the dot color.
# It only gave a text response because it could not use the CSV data directly.
#
# The CodeAgent tried to create the scatter plot, but it could not import
# matplotlib. It also had trouble accessing the data through csv_manager.
#
# This shows that a ToolCallingAgent is better for simple tasks that match
# existing tools, while a CodeAgent can be more useful for tasks that need
# custom code and data analysis.

# ========================================================================
# Q9


print('====== Q9===========================================================')


# 1. A ToolCallingAgent would be better for a task like loading a CSV
# and getting its columns. The task is simple and matches a specific tool.
#
# 2. A risk of a CodeAgent is that it can generate and run code that may
# cause errors or do something unexpected. A ToolCallingAgent only uses
# the tools that the developer gives it.
#
# 3. In summary, use a ToolCallingAgent for straightforward tasks with existing tools,
# and use a CodeAgent when you need more flexibility and custom code.
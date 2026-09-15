# JarirTech
# Warmup 07

from pathlib import Path
from datetime import datetime
import json
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dotenv import load_dotenv
from openai import OpenAI

from smolagents import (
    ToolCallingAgent,
    CodeAgent,
    OpenAIServerModel,
    tool,
)


# ============================================================
# SETUP


if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

client = OpenAI()

print("OpenAI client created.")
print("Warmup 07 started")


# ================================================================================

# Q1

print("\n========= Q1 ===============================================")


def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""

    fahrenheit = (celsius * 9 / 5) + 32

    return f"{celsius}°C is {fahrenheit}°F"


# JSON schema describing the function to the LLM.
# This follows the same function-tool structure used in the lesson.

celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "Temperature in degrees Celsius.",
                }
            },
            "required": ["celsius"],
        },
    },
}


print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))


# ============================================================================

# Q2


print("\n========= Q2 ===============================================")


def get_current_time() -> str:
    """Return the current local time as a formatted string."""

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


get_current_time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Returns the current local time as a string.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


# Q2 uses only the time tool.
tools = [
    get_current_time_schema
]


def run_agent(user_prompt: str) -> str:
    """Run a minimal ReAct-style agent for a single user prompt."""

    SYSTEM_PROMPT = (
        "You are a simple assistant that can tell the current time. "
        "Use the tool get_current_time whenever a user asks about the time."
    )

    # Step 1: start the conversation.
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    # Step 2: first API call.
    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    print("First response received from model...")

    first_message = first_response.choices[0].message

    assistant_entry = {
        "role": "assistant",
        "content": first_message.content,
    }

    if first_message.tool_calls:
        assistant_entry["tool_calls"] = [
            tool_call.model_dump()
            for tool_call in first_message.tool_calls
        ]

    messages.append(assistant_entry)

    # Step 3: check if a tool was requested.
    if first_message.tool_calls:

        print("Agentic mode engaged...")

        for tool_call in first_message.tool_calls:

            function_name = tool_call.function.name

            if function_name == "get_current_time":
                tool_result = get_current_time()
            else:
                tool_result = (
                    f"Error: unknown tool {function_name}."
                )

            print("Tool called:", function_name)
            print("Tool result:", tool_result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )

        # Step 4: second API call after tool result.
        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )

        print("Second response received from model...")

        final_message = second_response.choices[0].message

        return final_message.content or ""

    print("No tools needed.")

    return first_message.content or ""


# ------------------------------------------------------------
# Prediction
#
# 1. Will run_agent(
#    "Convert 100 degrees Celsius to Fahrenheit"
#    ) trigger a tool call?
#
#    No. The only available tool is get_current_time, and the
#    question is about temperature conversion, not time.
#
# 2. How many API calls will be made?
#
#    I predict one API call because no tool should be needed.
#    The first model response should be the final answer.
# ------------------------------------------------------------

response = run_agent(
    "Convert 100 degrees Celsius to Fahrenheit"
)

print("Result:", response)

# Prediction check:
# My prediction was correct if no tool was called and only one
# API call was needed. The model could answer the temperature
# conversion without using get_current_time.


# ============================================================

# Q3

print("\n========= Q3 ===============================================")


# Q3 now gives the model both tools.

tools = [
    get_current_time_schema,
    celsius_to_fahrenheit_schema,
]


def run_agent(user_prompt: str) -> str:
    """Run a ReAct-style agent with both available tools."""

    SYSTEM_PROMPT = (
        "You are a helpful assistant. "
        "Use the available tools when they are needed."
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    first_message = first_response.choices[0].message

    assistant_entry = {
        "role": "assistant",
        "content": first_message.content,
    }

    if first_message.tool_calls:
        assistant_entry["tool_calls"] = [
            tool_call.model_dump()
            for tool_call in first_message.tool_calls
        ]

    messages.append(assistant_entry)

    if first_message.tool_calls:

        print("Agentic mode engaged...")

        for tool_call in first_message.tool_calls:

            function_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments or "{}"
            )

            if function_name == "get_current_time":

                tool_result = get_current_time()

            elif function_name == "celsius_to_fahrenheit":

                tool_result = celsius_to_fahrenheit(
                    arguments["celsius"]
                )

            else:

                tool_result = (
                    f"Error: unknown tool {function_name}."
                )

            print("Tool called:", function_name)
            print("Tool result:", tool_result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )

        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )

        final_message = second_response.choices[0].message

        return final_message.content or ""

    return first_message.content or ""


response_a = run_agent(
    "What is 37 degrees Celsius in Fahrenheit?"
)

print("Response A:", response_a)
# A tool was called. The agent used celsius_to_fahrenheit
# because the question directly asked for a temperature
# conversion.


response_b = run_agent(
    "What is the boiling point of water in plain English?"
)

print("Response B:", response_b)
# No tool was called. This was a general knowledge question,
# so neither available tool was needed.


# ===============================================================================

# Q4



print("\n========= Q4 ===============================================")



RESOURCES_DIR = Path("resources")


class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- helpers --------------------------------------

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
                if p.is_file()
                and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:

            files = self._available_csv_files()

            example = (
                files[0]
                if files
                else "your_file.csv"
            )

            return {
                "error": (
                    "No CSV is loaded yet. "
                    "First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }

        return None

    # --- Tools --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """

        files = self._available_csv_files()

        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one "
                    "or more .csv files inside it."
                ),
                "files": [],
            }

        return {
            "files": files
        }

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """

        filename = self._normalize_csv_name(
            filename
        )

        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": (
                    f"Could not find '{filename}' "
                    "in resources/."
                ),
                "available_files":
                    self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": (
                f"Loaded {filename} "
                f"with shape {self.df.shape}."
            ),
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

    def summarize_columns(
        self,
        columns: list[str] | None = None
    ):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all").
        """

        error = self._ensure_loaded()

        if error:
            return error

        if columns is None:

            data = self.df

        else:

            missing = [
                column
                for column in columns
                if column not in self.df.columns
            ]

            if missing:
                return {
                    "error": (
                        "These columns are not in "
                        f"the data: {missing}"
                    )
                }

            data = self.df[columns]

        summary = (
            data
            .describe(include="all")
            .transpose()
            .round(3)
        )

        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """

        error = self._ensure_loaded()

        if error:
            return error

        if column not in self.df.columns:
            return {
                "error": (
                    f"'{column}' is not a column. "
                    f"Options: {self.df.columns.tolist()}"
                )
            }

        series = self.df[column]

        summary = series.describe().to_dict()

        cleaned = {}

        for key, value in summary.items():

            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(
        self,
        y: str,
        x: str | None = None,
        plot_type: str = "line"
    ):
        """
        Plot from the active CSV.

        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """

        error = self._ensure_loaded()

        if error:
            return error

        if plot_type not in [
            "scatter",
            "line",
        ]:
            return (
                "Error: I can only do "
                "'scatter' or 'line'."
            )

        if y not in self.df.columns:
            return (
                f"Error: column '{y}' is not in "
                f"{self.df.columns.tolist()}"
            )

        if x == y:
            x = None

        if (
            plot_type == "scatter"
            and x is None
        ):
            return (
                "Error: scatter plots need "
                "both x and y columns."
            )

        title_csv = (
            self.csv_name
            or "current CSV"
        )

        if x is None:

            ax = self.df[y].plot(
                kind="line"
            )

            ax.set_title(
                f"{title_csv} | "
                f"Line plot: {y} vs row index"
            )

            plt.show()

            return (
                f"Plotted {y} vs row index "
                "as a line plot."
            )

        if x not in self.df.columns:
            return (
                f"Error: column '{x}' is not in "
                f"{self.df.columns.tolist()}"
            )

        ax = self.df.plot(
            x=x,
            y=y,
            kind=plot_type,
        )

        ax.set_title(
            f"{title_csv} | "
            f"{plot_type.title()} plot: "
            f"{y} vs {x}"
        )

        plt.show()

        return (
            f"Plotted {y} vs {x} "
            f"as a {plot_type}."
        )

    # --------------------------------------------------------
   

    def compute_correlation(
        self,
        col1: str,
        col2: str
    ):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """

        error = self._ensure_loaded()

        if error:
            return error

        if col1 not in self.df.columns:
            return {
                "error": (
                    f"Column '{col1}' not found."
                )
            }

        if col2 not in self.df.columns:
            return {
                "error": (
                    f"Column '{col2}' not found."
                )
            }

        try:

            data = self.df[
                [col1, col2]
            ].dropna()

            r, p = pearsonr(
                data[col1],
                data[col2]
            )

            return {
                "col1": col1,
                "col2": col2,
                "pearson_r": round(
                    float(r),
                    4
                ),
                "p_value": round(
                    float(p),
                    4
                ),
            }

        except Exception as error:

            return {
                "error": (
                    "Could not compute correlation: "
                    f"{error}"
                )
            }


print("CsvManager class defined")


csv_manager = CsvManager(
    RESOURCES_DIR
)


# ------------------------------------------------------------
# node_tools + compute_correlation
# ------------------------------------------------------------

node_tools = {
    "list_csv_files":
        csv_manager.list_csv_files,

    "load_csv":
        csv_manager.load_csv,

    "get_columns":
        csv_manager.get_columns,

    "summarize_columns":
        csv_manager.summarize_columns,

    "describe_column":
        csv_manager.describe_column,

    "plot_data":
        csv_manager.plot_data,

    "compute_correlation":
        csv_manager.compute_correlation,
}


# ------------------------------------------------------------
# tools_schema  + compute_correlation
# ------------------------------------------------------------

tools_schema = [

    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": (
                "List available CSV files "
                "in the resources/ folder."
            ),
        },
    },

    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": (
                "Load a CSV file from the "
                "resources/ folder and make "
                "it the active dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": (
                            "CSV filename in resources/, "
                            "e.g. 'bike_commute.csv'."
                        ),
                    }
                },
                "required": ["filename"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": (
                "Get the column names of "
                "the currently loaded CSV."
            ),
        },
    },

    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": (
                "Show basic summary statistics "
                "for columns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": (
                            "Optional list of column names. "
                            "If omitted, summarize all columns."
                        ),
                    }
                },
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": (
                "Show basic summary statistics "
                "for a single column."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": (
                            "Column name to describe."
                        ),
                    }
                },
                "required": ["column"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": (
                "Plot data from the active CSV. "
                "If only y is provided, plot y "
                "vs row index."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {
                        "type": "string",
                        "description":
                            "Column name for y-axis.",
                    },
                    "x": {
                        "type": "string",
                        "description":
                            "Optional column name for x-axis.",
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": [
                            "scatter",
                            "line",
                        ],
                        "description":
                            "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },

    # New Q4 tool.
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": (
                "Compute Pearson correlation "
                "between two numeric columns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description":
                            "First numeric column.",
                    },
                    "col2": {
                        "type": "string",
                        "description":
                            "Second numeric column.",
                    },
                },
                "required": [
                    "col1",
                    "col2",
                ],
            },
        },
    },
]


print(
    "compute_correlation added to "
    "tools_schema and node_tools."
)


# ============================================================

# Q5


print("\n========= Q5 ===============================================")


def run_agent_cycle(
    messages,
    user_text,
    max_tool_rounds=5
):
    """
    Run through one ReAct-agent loop.

    REASON:
        Ask the model what to do.

    ACT:
        Run requested tools.

    OBSERVE:
        Add the tool result back to the conversation.

    Args:
        messages: Conversation history.
        user_text: Current user question.
        max_tool_rounds: Maximum tool rounds.

    Returns:
        Final assistant response.
    """

    messages.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    def observe_tool_result(
        tool_call_id,
        result
    ):
        content = (
            json.dumps(
                result,
                default=str
            )
            if not isinstance(result, str)
            else result
        )

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    for loop_idx in range(
        max_tool_rounds
    ):

        # REASON
        response = (
            client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                tools=tools_schema,
            )
        )

        msg = (
            response
            .choices[0]
            .message
        )

        assistant_entry = {
            "role": "assistant",
            "content": msg.content,
        }

        if msg.tool_calls:
            assistant_entry[
                "tool_calls"
            ] = [
                tool_call.model_dump()
                for tool_call
                in msg.tool_calls
            ]

        messages.append(
            assistant_entry
        )

        # No tool calls = final answer.
        if not msg.tool_calls:
            return msg.content

        # ACT + OBSERVE
        for tool_call in msg.tool_calls:

            name = (
                tool_call
                .function
                .name
            )

            tool_args = json.loads(
                tool_call
                .function
                .arguments
                or "{}"
            )

            print(
                f"ACT: {name}({tool_args})"
            )

            fn = node_tools.get(
                name
            )

            if fn is None:

                result = {
                    "error": (
                        f"Tool '{name}' "
                        "not found."
                    )
                }

            else:

                try:

                    result = (
                        fn(**tool_args)
                        if tool_args
                        else fn()
                    )

                except Exception as error:

                    print(
                        "Tool error in "
                        f"{name}: {error}"
                    )

                    result = {
                        "error": (
                            f"Tool '{name}' "
                            f"failed: {error}"
                        )
                    }

            print(
                "OBSERVE:",
                result
            )

            messages.append(
                observe_tool_result(
                    tool_call.id,
                    result
                )
            )

    return (
        "I hit the tool-round limit. "
        "Try a simpler request."
    )


SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)


messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
]


result = run_agent_cycle(
    messages,
    (
        "Load bike_commute.csv and compute the correlation "
        "between avg_traffic_density and avg_speed_kmh."
    )
)

print("Final response:")
print(result)

# The agent should now succeed because compute_correlation
# is available as a tool instead of hitting the tool-round
# limit from the original lesson example.


# ============================================================

# Q6


print("\n========= Q6 ===============================================")


# system:
# Gives instructions that control how the agent behaves.
#
# user:
# Contains the user's question or request.
#
# assistant:
# Contains the model response and any requested tool calls.
#
# tool:
# Contains the result returned by a tool. The model reads this
# result during the next REASON step of the ReAct loop.


print(
    json.dumps(
        messages,
        indent=2,
        default=str,
    )
)


# ============================================================

# Q7


print("\n========= Q7 ===============================================")


@tool
def compute_correlation(
    col1: str,
    col2: str
) -> dict:
    """Compute Pearson correlation between two columns.

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        A dictionary containing the column names, Pearson
        correlation coefficient, and p-value.
    """

    return csv_manager.compute_correlation(
        col1,
        col2,
    )


print(compute_correlation.description)

# Q7 Reflection:
#
# In Q4, I manually wrote the JSON schema with the tool name,
# description, parameter types, and required arguments.
#
# Smolagents creates this information automatically from the
# function name, type hints, and Google-style docstring.
#
# As the developer, I need to provide clear type hints and a
# good docstring that explains what the tool does, describes
# each argument, and explains what the function returns.


# ===========================================================================

# Q8


print("\n========= Q8 ===============================================")


# Wraping ALL tools from the lesson for smolagents.


@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dictionary containing the available CSV filenames.
    """

    return csv_manager.list_csv_files()


@tool
def load_csv(
    filename: str
) -> dict:
    """Load a CSV file and make it the active dataset.

    Args:
        filename: CSV filename in resources/. It may be given
            with or without the .csv extension.

    Returns:
        A dictionary with the load result and column names.
    """

    return csv_manager.load_csv(
        filename
    )


@tool
def get_columns() -> list | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dictionary if no
        CSV has been loaded.
    """

    return csv_manager.get_columns()


@tool
def summarize_columns(
    columns: list[str] | None = None
) -> dict:
    """Return summary statistics for selected columns.

    Args:
        columns: Optional list of column names. If None,
            summarize all columns.

    Returns:
        A dictionary containing summary statistics, or an
        error dictionary.
    """

    return csv_manager.summarize_columns(
        columns
    )


@tool
def describe_column(
    column: str
) -> dict:
    """Describe one column from the active CSV.

    Args:
        column: Name of the column to describe.

    Returns:
        A dictionary containing summary statistics, or an
        error dictionary.
    """

    return csv_manager.describe_column(
        column
    )


@tool
def plot_data(
    y: str,
    x: str | None = None,
    plot_type: str = "line"
) -> str | dict:
    """Plot data from the active CSV.

    Args:
        y: Column name for the y-axis.
        x: Optional column name for the x-axis.
        plot_type: Plot type. Use "line" or "scatter".

    Returns:
        A success message, or an error result.
    """

    return csv_manager.plot_data(
        y=y,
        x=x,
        plot_type=plot_type,
    )


# Both agents use the SAME list from the lesson,
# including the new correlation tool.

TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]


api_key = os.getenv(
    "OPENAI_API_KEY"
)


model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini",
)


TOOL_SYSTEM_PROMPT = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)


CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.

You can do two kinds of actions:
1) Call the provided tools.
2) Write and execute Python code when tools are not enough.

Rules:
- Prefer tools for simple tasks.
- IMPORTANT: If the user requests plot styling such as
  color, marker, title, labels, or grid that plot_data cannot
  control, DO NOT call plot_data for that styled plot.
- Instead, write matplotlib code directly so the plot matches
  the user's request.
- Be honest: only claim you did something if the code or tool
  actually did it.
- Assume the active dataset lives in csv_manager.df after a
  CSV is loaded.
"""


tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model,
    instructions=TOOL_SYSTEM_PROMPT,
)


code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=[
        "pandas",
        "matplotlib.pyplot",
        "numpy",
    ],
    max_steps=8,
)


prompt = (
    "Load bike_commute.csv. "
    "Plot avg_heart_rate vs duration_min "
    "as a scatter plot with green dots."
)


response_tool = tool_agent.run(
    prompt
)


response_code = code_agent.run(
    prompt,
    additional_args={
        "csv_manager": csv_manager
    },
)


print(
    "ToolCallingAgent Response:"
)
print(response_tool)

print(
    "\nCodeAgent Response:"
)
print(response_code)


# Q8 Reflection:
#
# The ToolCallingAgent loaded the CSV and used the
# plot_data tool to create a scatter plot.
#
# It said the dots were green, but plot_data does not have
# a color option, so the ToolCallingAgent did not actually
# control the dot color.
#
# The CodeAgent wrote matplotlib code and used:
# plt.scatter(x, y, color="green")
# so it was able to make the dots green.
#
# This shows that ToolCallingAgent is better when the
# existing tools already support the full task.
#
# CodeAgent is more useful when custom code or extra
# options, such as plot colors, are needed.

# ============================================================



print("\n========= Q9 ===============================================")


# Q9 Reflection:
#
# 1. A ToolCallingAgent is better for tasks like loading
#    a CSV or getting column names because those tasks
#    already have clear tools.
#
# 2. A CodeAgent can generate and run Python code.
#    One risk is that the generated code may contain
#    errors or perform something unexpected.
# JarirTech
# Warmup 07

from pathlib import Path
import json
import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dotenv import load_dotenv
from openai import OpenAI

from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel, tool


# ============================================================
# .env SETUP
# ============================================================

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

client = OpenAI()
print("OpenAI client created.")

print("Warmup 07 started")


# ============================================================
# Q1: Celsius to Fahrenheit Tool
# ============================================================

print("--------- Q1 -----------------------------------------------")


def celsius_to_fahrenheit(celsius: float) -> str:
    """
    Convert a temperature from Celsius to Fahrenheit.

    Args:
        celsius: Temperature in degrees Celsius.

    Returns:
        A formatted string showing the Celsius and Fahrenheit
        temperatures.
    """

    fahrenheit = (celsius * 9 / 5) + 32

    return f"{celsius}°C is {fahrenheit}°F"


celsius_to_fahrenheit_schema = {
    "name": "celsius_to_fahrenheit",
    "description": "Convert a temperature from Celsius to Fahrenheit.",
    "parameters": {
        "type": "object",
        "properties": {
            "celsius": {
                "type": "number",
                "description": "Temperature in degrees Celsius."
            }
        },
        "required": ["celsius"]
    }
}


celsius_to_fahrenheit_openai_tool = {
    "type": "function",
    "function": celsius_to_fahrenheit_schema
}


print(
    f"celsius_to_fahrenheit(0) = "
    f"{celsius_to_fahrenheit(0)}"
)

print(
    f"celsius_to_fahrenheit(100) = "
    f"{celsius_to_fahrenheit(100)}"
)

print(
    f"celsius_to_fahrenheit(-40) = "
    f"{celsius_to_fahrenheit(-40)}"
)


# ============================================================
# Q2: Simple ReAct-Style Agent
# ============================================================

print("\n--------- Q2 -----------------------------------------------")


def get_current_time() -> str:
    """
    Return the current local time as a formatted string.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


get_current_time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Returns the current local time as a string.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}


tools = [
    get_current_time_schema
]


def run_agent(user_prompt: str) -> str:
    """
    Run a minimal ReAct-style agent for one user prompt.
    """

    system_prompt = """
    You are a simple assistant that can tell the current time.
    Use the tool get_current_time whenever a user asks about the time.
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    print("First response received from model...")

    first_message = first_response.choices[0].message

    assistant_entry = {
        "role": "assistant",
        "content": first_message.content
    }

    if first_message.tool_calls:
        assistant_entry["tool_calls"] = [
            tc.model_dump()
            for tc in first_message.tool_calls
        ]

    messages.append(assistant_entry)

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
                    "content": tool_result
                }
            )

        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages
        )

        print("Second response received from model...")

        final_message = second_response.choices[0].message

        return final_message.content or ""

    print("No tools needed.")

    return first_message.content or ""


# Prediction:
# The model should NOT call get_current_time because the
# question is about temperature conversion, not time.
# I expect one API call because no tool should be needed.

response = run_agent(
    "Convert 100 degrees Celsius to Fahrenheit."
)

print("Result:", response)


# ============================================================
# Q3: Multi-Tool ReAct Agent
# ============================================================

print("\n--------- Q3 -----------------------------------------------")


tools = [
    get_current_time_schema,
    celsius_to_fahrenheit_openai_tool
]


def run_agent(user_prompt: str) -> str:
    """
    Run a ReAct-style agent with two available tools.
    """

    system_prompt = """
    You are a helpful assistant.
    Use tools when they are needed.
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    first_message = first_response.choices[0].message

    assistant_message = {
        "role": "assistant",
        "content": first_message.content
    }

    if first_message.tool_calls:
        assistant_message["tool_calls"] = [
            tc.model_dump()
            for tc in first_message.tool_calls
        ]

    messages.append(assistant_message)

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
                    "content": tool_result
                }
            )

        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages
        )

        final_message = second_response.choices[0].message

        return final_message.content or ""

    return first_message.content or ""


response_a = run_agent(
    "What is 37 degrees Celsius in Fahrenheit?"
)

print("Response A:", response_a)

# The celsius_to_fahrenheit tool was called because the
# question specifically required a temperature conversion.


response_b = run_agent(
    "What is the boiling point of water in plain English?"
)

print("Response B:", response_b)

# No tool was needed because this was a general knowledge
# question and neither available tool was relevant.


# ============================================================
# Q4: Multi-Tool CSV Agent
# ============================================================

print("\n--------- Q4 -----------------------------------------------")


class CsvManager:
    """
    Manage CSV files and perform basic data analysis.
    """

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

    # --- Public tools --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """

        files = self._available_csv_files()

        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more "
                    ".csv files inside it."
                ),
                "files": [],
            }

        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        Args:
            filename: CSV filename with or without .csv.

        Returns:
            A dictionary containing the load status and columns.
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

    def summarize_columns(
        self,
        columns: list[str] | None = None
    ):
        """
        Return basic summary statistics for one or more columns.

        If columns is None, summarize all columns.
        """

        error = self._ensure_loaded()

        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [
                c for c in columns
                if c not in self.df.columns
            ]

            if missing:
                return {
                    "error": (
                        f"These columns are not in the data: {missing}"
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
        Return basic summary statistics for a single column.
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

        s = self.df[column]
        summary = s.describe().to_dict()

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

        If x is None, plot y versus row index.
        If x is provided, plot y versus x.
        """

        error = self._ensure_loaded()

        if error:
            return error

        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."

        if y not in self.df.columns:
            return (
                f"Error: column '{y}' is not in "
                f"{self.df.columns.tolist()}"
            )

        if x == y:
            x = None

        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."

        title_csv = self.csv_name or "current CSV"

        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(
                f"{title_csv} | Line plot: {y} vs row index"
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
            kind=plot_type
        )

        ax.set_title(
            f"{title_csv} | {plot_type.title()} plot: "
            f"{y} vs {x}"
        )

        plt.show()

        return (
            f"Plotted {y} vs {x} "
            f"as a {plot_type}."
        )

    def compute_correlation(
        self,
        col1: str,
        col2: str
    ):
        """
        Compute Pearson correlation between two numeric columns.

        Args:
            col1: First numeric column.
            col2: Second numeric column.

        Returns:
            A dictionary containing Pearson r and p-value,
            or an error dictionary.
        """

        error = self._ensure_loaded()

        if error:
            return error

        if col1 not in self.df.columns:
            return {
                "error": f"Column '{col1}' not found."
            }

        if col2 not in self.df.columns:
            return {
                "error": f"Column '{col2}' not found."
            }

        try:
            r, p = pearsonr(
                self.df[col1],
                self.df[col2]
            )

        except Exception as error:
            return {
                "error": (
                    "Could not compute correlation: "
                    f"{error}"
                )
            }

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4),
        }


print("Class defined")


# ------------------------------------------------------------
# Resources path and CSV manager
# ------------------------------------------------------------

RESOURCES_DIR = Path("resources")

csv_backend = CsvManager(RESOURCES_DIR)

print("CsvManager created")


# ------------------------------------------------------------
# Node tools
# ------------------------------------------------------------

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "plot_data": csv_backend.plot_data,
    "compute_correlation": csv_backend.compute_correlation,
}


# ------------------------------------------------------------
# Tool schemas
# ------------------------------------------------------------

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": (
                "List available CSV files in the resources/ folder."
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": (
                "Load a CSV file from the resources/ folder "
                "and make it the active dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": (
                            "CSV filename in resources/, "
                            "for example 'bike_commute.csv'."
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
                "Get the column names of the currently loaded CSV."
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": (
                "Show basic summary statistics for columns "
                "using pandas.describe."
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
                "Show basic summary statistics for a single column."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
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
                "If only y is provided, plot y versus row index."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {
                        "type": "string",
                        "description": "Column name for y-axis.",
                    },
                    "x": {
                        "type": "string",
                        "description": "Optional column name for x-axis.",
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": (
                            "Type of plot to create."
                        ),
                    },
                },
                "required": ["y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": (
                "Compute the Pearson correlation coefficient "
                "between two numeric columns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "First numeric column.",
                    },
                    "col2": {
                        "type": "string",
                        "description": "Second numeric column.",
                    },
                },
                "required": ["col1", "col2"],
            },
        },
    },
]


print("CSV tools added successfully.")


# ============================================================
# Q5: Agent Cycle
# ============================================================

print("\n--------- Q5 -----------------------------------------------")


def run_agent_cycle(
    messages,
    user_text,
    max_tool_rounds=5
):
    """
    Run through one ReAct agent loop using the CSV tools.
    """

    messages.append(
        {
            "role": "user",
            "content": user_text
        }
    )

    def observe_tool_result(
        tool_call_id,
        result
    ):
        """
        Return a tool result as a message for the LLM.
        """

        content = (
            json.dumps(result, default=str)
            if not isinstance(result, str)
            else result
        )

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    for loop_idx in range(max_tool_rounds):

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        assistant_entry = {
            "role": "assistant",
            "content": msg.content,
        }

        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                tc.model_dump()
                for tc in msg.tool_calls
            ]

        messages.append(assistant_entry)

        if not msg.tool_calls:
            return msg.content

        for tool_call in msg.tool_calls:

            name = tool_call.function.name

            tool_args = json.loads(
                tool_call.function.arguments or "{}"
            )

            print(
                f"ACT: {name}({tool_args})"
            )

            fn = node_tools.get(name)

            if fn is None:

                result = {
                    "error": (
                        f"Tool '{name}' not found."
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
                        f"Tool error in {name}: "
                        f"{type(error).__name__}: {error}"
                    )

                    result = {
                        "error": (
                            f"Tool '{name}' failed: "
                            f"{type(error).__name__}: {error}"
                        )
                    }

            print("OBSERVE:", result)

            messages.append(
                observe_tool_result(
                    tool_call.id,
                    result
                )
            )

    return "I hit the tool-round limit. Try a simpler request."


SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first "
    "(or list available CSV files). "
    "Keep answers short and student-friendly."
)


messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]


result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation "
    "between avg_traffic_density and avg_speed_kmh."
)

print("Final answer:")
print(result)


# ============================================================
# Q6: Message Roles
# ============================================================

print("\n--------- Q6 -----------------------------------------------")


# system:
# Gives the agent instructions about how to behave.

# user:
# Contains the user's question or request.

# assistant:
# Contains the model's response and any requested tool calls.

# tool:
# Contains the result returned by a tool after it is executed.


print(
    json.dumps(
        messages,
        indent=2,
        default=str
    )
)


# ============================================================
# Q7: Smolagents @tool
# ============================================================

print("\n--------- Q7 -----------------------------------------------")


@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dictionary with a "files" list, or a message if none exist.
    """
    return csv_backend.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it active.

    Args:
        filename: CSV filename in resources/.

    Returns:
        A dictionary containing load status and column names.
    """
    return csv_backend.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dictionary.
    """
    return csv_backend.get_columns()


@tool
def summarize_columns(
    columns: list[str] | None = None
) -> dict:
    """Return summary statistics for selected columns.

    Args:
        columns: Optional list of column names. If None,
            summarize all columns.

    Returns:
        A dictionary of summary statistics.
    """
    return csv_backend.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column using basic statistics.

    Args:
        column: Name of the column to describe.

    Returns:
        A dictionary of basic statistics.
    """
    return csv_backend.describe_column(column)


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
        plot_type: "line" or "scatter".

    Returns:
        A short success message or an error.
    """
    return csv_backend.plot_data(
        y=y,
        x=x,
        plot_type=plot_type
    )


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
        A dictionary containing Pearson r and p-value,
        or an error dictionary.
    """
    return csv_backend.compute_correlation(
        col1,
        col2
    )


print("Smolagents tools created successfully.")


# ============================================================
# Q8: ToolCallingAgent vs CodeAgent
# ============================================================

print("\n--------- Q8 -----------------------------------------------")


model = OpenAIServerModel(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_id="gpt-4o-mini",
)


# IMPORTANT:
# Both agents use the SAME complete TOOLS list from the lesson.
TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]


SYSTEM_PROMPT = (
    "You are a small data assistant to help analyze files "
    "stored in resources/. "
    "Use the available tools to do any work requested "
    "(do not guess). "
    "Keep answers short and student-friendly."
)


tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model,
    instructions=SYSTEM_PROMPT,
)


CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.

You can do two kinds of actions:
1) Call the provided tools.
2) Write and execute Python code when tools are not enough.

Rules:
- Prefer tools for simple tasks.
- IMPORTANT: If the user requests plot styling
  (color, marker, title text, labels, grid, etc.)
  that the plot_data tool cannot control, DO NOT call plot_data.
  Instead, write matplotlib code directly so the plot matches
  the request.
- If code execution fails, do not fall back to plot_data when
  the user requested styling like color.
- Explain what failed and what you would need to proceed.
- Be honest: only claim you did something if the code or tool
  actually did it.
- Assume the active dataset lives in csv_manager.df after
  a CSV has been loaded.
"""


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


prompt = """
Load bike_commute.csv, and plot avg_heart_rate versus
duration_min as a scatter plot with green dots.
"""


# ToolCallingAgent test
response_tool = tool_agent.run(prompt)

# CodeAgent test
response_code = code_agent.run(
    prompt,
    additional_args={
        "csv_manager": csv_backend
    },
)


print("ToolCallingAgent Response:")
print(response_tool)

print("\nCodeAgent Response:")
print(response_code)


# ------------------------------------------------------------
# Reflection:
#
# The ToolCallingAgent can only use the tools provided to it.
# It has the complete CSV toolset, including load_csv,
# plot_data, and compute_correlation.
#
# However, plot_data does not provide an option to control
# point color, so the ToolCallingAgent cannot directly add
# green dots through that tool.
#
# The CodeAgent receives the same TOOLS list, but it can also
# write and execute Python code when the available tools are
# not enough.
#
# Therefore, the CodeAgent can create a custom matplotlib
# plot with green dots.
# ------------------------------------------------------------


# ============================================================
# Q9: Final Reflection
# ============================================================

print("\n--------- Q9 -----------------------------------------------")


# ------------------------------------------------------------
# Reflection:
#
# 1. A ToolCallingAgent is best when the needed tools already
#    exist, such as loading a CSV, describing columns,
#    plotting data, or computing a correlation.
#
# 2. A CodeAgent is more flexible because it can use the
#    provided tools and also create and run Python code when
#    the existing tools are not enough.
#
# 3. ToolCallingAgents are more limited because they can only
#    perform actions supported by their available tools.
#
# 4. CodeAgents are useful for flexible tasks that require
#    custom Python code, such as customized matplotlib plots.
#
# 5. CodeAgents are more powerful, but generated code can
#    sometimes contain errors, so their results should still
#    be checked.

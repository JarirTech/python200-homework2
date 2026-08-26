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

from smolagents import (
    ToolCallingAgent,
    CodeAgent,
    OpenAIServerModel,
    tool
)


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


# Standalone function schema.
# This follows the lesson format.
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

    Returns:
        str: The current local date and time.
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

    Args:
        user_prompt: User question or request.

    Returns:
        str: The final response from the model.
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

    # First API call.
    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    print("First response received from model...")

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

    # Check if the model requested a tool.
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

        # Second API call after receiving the tool result.
        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages
        )

        print("Second response received from model...")

        final_message = second_response.choices[0].message

        return final_message.content or ""

    print("No tools needed.")

    return first_message.content or ""


# ------------------------------------------------------------
# Prediction before calling run_agent()
#
# Prediction:
# 1. The model should NOT call get_current_time because the
#    question asks for a temperature conversion, not the time.
#
# 2. I expect one API call because no tool should be needed.
# ------------------------------------------------------------

response = run_agent(
    "Convert 100 degrees Celsius to Fahrenheit."
)

print("Result:", response)

# No tool was called because the question did not ask for
# the current time. The model answered the temperature
# conversion directly, so only one API call was needed.


# ============================================================
# Q3: Multi-Tool ReAct Agent
# ============================================================

print("\n--------- Q3 -----------------------------------------------")


# OpenAI API tool format for the Celsius conversion tool.
celsius_to_fahrenheit_openai_tool = {
    "type": "function",
    "function": celsius_to_fahrenheit_schema
}


tools = [
    get_current_time_schema,
    celsius_to_fahrenheit_openai_tool
]


def run_agent(user_prompt: str) -> str:
    """
    Run a ReAct-style agent with two available tools.

    Args:
        user_prompt: User question or request.

    Returns:
        str: The final response from the model.
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

    # First API call.
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

    # Check for tool calls.
    if first_message.tool_calls:

        print("Agentic mode engaged...")

        for tool_call in first_message.tool_calls:

            function_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
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

        # Second API call.
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

# A tool was called because the question required a
# Celsius-to-Fahrenheit conversion, and that calculation
# was available through the celsius_to_fahrenheit tool.


response_b = run_agent(
    "What is the boiling point of water in plain English?"
)

print("Response B:", response_b)

# No tool was called because the question was general
# knowledge and neither available tool was needed.


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

    # --------------------------------------------------------
    # Internal helper methods
    # --------------------------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        """
        Add the .csv extension when it is missing.

        Args:
            filename: CSV filename.

        Returns:
            str: Normalized CSV filename.
        """

        if not filename.lower().endswith(".csv"):
            return filename + ".csv"

        return filename

    def _available_csv_files(self) -> list[str]:
        """
        Find CSV files in the resources directory.

        Returns:
            list[str]: Available CSV filenames.
        """

        if not self.resources_dir.exists():
            return []

        return sorted(
            [
                path.name
                for path in self.resources_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        """
        Check whether a CSV file has been loaded.

        Returns:
            dict or None: Error dictionary when no CSV is
            loaded, otherwise None.
        """

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

    # --------------------------------------------------------
    # Public tools
    # --------------------------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in the resources directory.

        Returns:
            dict: Available CSV filenames or a message when
            no CSV files are found.
        """

        files = self._available_csv_files()

        if not files:

            return {
                "message": (
                    "No CSV files found in resources/."
                ),
                "files": []
            }

        return {
            "files": files
        }

    def load_csv(self, filename: str):
        """
        Load a CSV file from the resources directory.

        Args:
            filename: CSV filename with or without .csv.

        Returns:
            dict: Information about the loaded CSV, including
            its shape and columns, or an error message.
        """

        filename = self._normalize_csv_name(filename)

        path = self.resources_dir / filename

        if not path.exists():

            return {
                "error": (
                    f"Could not find '{filename}' "
                    "in resources/."
                ),
                "available_files":
                    self._available_csv_files()
            }

        self.df = pd.read_csv(path)

        self.csv_name = filename

        return {
            "message": (
                f"Loaded {filename} "
                f"with shape {self.df.shape}."
            ),
            "columns": self.df.columns.tolist()
        }

    def get_columns(self):
        """
        Return column names for the active CSV.

        Returns:
            list or dict: Column names when a CSV is loaded,
            otherwise an error dictionary.
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
        Return summary statistics for selected columns.

        Args:
            columns: Optional list of columns to summarize.
                If None, summarize all columns.

        Returns:
            dict: Summary statistics or an error dictionary.
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
                        f"These columns are not in "
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
        Return summary statistics for one column.

        Args:
            column: Column to describe.

        Returns:
            dict: Descriptive statistics for the column,
            or an error dictionary.
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

        summary = self.df[column].describe().to_dict()

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
        Plot data from the active CSV.

        Args:
            y: Column to use for the y-axis.
            x: Optional column to use for the x-axis.
            plot_type: Plot type, either line or scatter.

        Returns:
            str or dict: Plot description or an error message.
        """

        error = self._ensure_loaded()

        if error:
            return error

        if plot_type not in ["scatter", "line"]:

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

        if plot_type == "scatter" and x is None:

            return (
                "Error: scatter plots need "
                "both x and y columns."
            )

        title_csv = self.csv_name or "current CSV"

        if x is None:

            ax = self.df[y].plot(kind="line")

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
            kind=plot_type
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
            dict: Column names, Pearson r, and p-value,
            or an error dictionary.
        """

        if self.df is None:

            return {
                "error": (
                    "No CSV is loaded yet. "
                    "Load a CSV first."
                )
            }

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
            "p_value": round(float(p), 4)
        }


print("CsvManager class defined")


# ------------------------------------------------------------
# Resources path
# ------------------------------------------------------------

resources_dir = Path("resources")

csv_manager = CsvManager(resources_dir)

print("CsvManager created")


# ------------------------------------------------------------
# Tool schemas
# ------------------------------------------------------------

tools_schema = [

    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the CSV file."
                    }
                },
                "required": ["filename"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get column names from the loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Summarize columns in the loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": []
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Describe one column in the loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string"
                    }
                },
                "required": ["column"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Create a line or scatter plot from the loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {
                        "type": "string"
                    },
                    "x": {
                        "type": "string"
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": ["line", "scatter"]
                    }
                },
                "required": ["y"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute Pearson correlation between two columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string"
                    },
                    "col2": {
                        "type": "string"
                    }
                },
                "required": ["col1", "col2"]
            }
        }
    }
]


# ------------------------------------------------------------
# Node tools
# ------------------------------------------------------------

node_tools = {
    "list_csv_files": csv_manager.list_csv_files,
    "load_csv": csv_manager.load_csv,
    "get_columns": csv_manager.get_columns,
    "summarize_columns": csv_manager.summarize_columns,
    "describe_column": csv_manager.describe_column,
    "plot_data": csv_manager.plot_data,
    "compute_correlation": csv_manager.compute_correlation
}


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
    Run the agent through repeated tool-use cycles.

    Args:
        messages: Conversation message history.
        user_text: New user request.
        max_tool_rounds: Maximum number of tool-use rounds.

    Returns:
        str: Final agent response or a tool-round limit message.
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

        content = (
            json.dumps(result, default=str)
            if not isinstance(result, str)
            else result
        )

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        }

    for loop_idx in range(max_tool_rounds):

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        assistant_entry = {
            "role": "assistant",
            "content": msg.content
        }

        if msg.tool_calls:

            assistant_entry["tool_calls"] = [
                tc.model_dump()
                for tc in msg.tool_calls
            ]

        messages.append(assistant_entry)

        # No tool calls means this is the final answer.
        if not msg.tool_calls:

            return msg.content

        # ACT + OBSERVE
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

                    result = {
                        "error": (
                            f"Tool failed: {error}"
                        )
                    }

            print("OBSERVE:", result)

            messages.append(
                observe_tool_result(
                    tool_call.id,
                    result
                )
            )

    return "I hit the tool-round limit."


system_prompt = (
    "You are a small data assistant for CSV files "
    "stored in resources/. "
    "Use the available tools to do data work. "
    "Do not guess. "
    "If no CSV is loaded, load one first. "
    "Keep answers short and student-friendly."
)


messages = [
    {
        "role": "system",
        "content": system_prompt
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
#
# user:
# Contains the user's question or request.
#
# assistant:
# Contains the model's response and any requested tool calls.
#
# tool:
# Contains the result returned by a tool after it is executed.


print(json.dumps(
    messages,
    indent=2,
    default=str
))


# ============================================================
# Q7: Smolagents @tool
# ============================================================

print("\n--------- Q7 -----------------------------------------------")


@tool
def list_csv_files() -> dict:
    """
    List available CSV files.

    Returns:
        dict: Available CSV filenames.
    """

    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """
    Load a CSV file.

    Args:
        filename: Name of the CSV file.

    Returns:
        dict: Information about the loaded CSV.
    """

    return csv_manager.load_csv(filename)


@tool
def get_columns() -> list:
    """
    Get column names from the loaded CSV.

    Returns:
        list: Column names.
    """

    return csv_manager.get_columns()


@tool
def summarize_columns(
    columns: list[str] | None = None
) -> dict:
    """
    Summarize columns in the loaded CSV.

    Args:
        columns: Optional list of columns to summarize.

    Returns:
        dict: Summary statistics.
    """

    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """
    Describe one column in the loaded CSV.

    Args:
        column: Name of the column.

    Returns:
        dict: Descriptive statistics.
    """

    return csv_manager.describe_column(column)


@tool
def plot_data(
    y: str,
    x: str | None = None,
    plot_type: str = "line"
) -> str:
    """
    Create a line or scatter plot from the loaded CSV.

    Args:
        y: Column used for the y-axis.
        x: Optional column used for the x-axis.
        plot_type: Plot type, either line or scatter.

    Returns:
        str: Description of the plot result.
    """

    return csv_manager.plot_data(
        y=y,
        x=x,
        plot_type=plot_type
    )


@tool
def compute_correlation(
    col1: str,
    col2: str
) -> dict:
    """
    Compute Pearson correlation between two columns.

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        dict: Column names, Pearson r, and p-value.
    """

    return csv_manager.compute_correlation(
        col1,
        col2
    )


print(compute_correlation.description)


# ------------------------------------------------------------
# Reflection:
#
# Smolagents creates the tool description from the function
# name, type hints, and docstring.
#
# The manual JSON schema in Q4 requires more code.
# Using @tool is simpler because Smolagents builds the tool
# information automatically.
#
# Good docstrings are important because they help the agent
# understand how and when to use a tool.
# ------------------------------------------------------------


# ============================================================
# Q8: ToolCallingAgent vs CodeAgent
# ============================================================

print("\n--------- Q8 -----------------------------------------------")


model = OpenAIServerModel(
    model_id="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)


# Both agents receive the SAME TOOLS list.
# The new compute_correlation tool is included with the
# existing CSV tools from the lesson.

TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation
]


tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model
)


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
    prompt
)


print("ToolCallingAgent Response:")
print(response_tool)

print("\nCodeAgent Response:")
print(response_code)


# ------------------------------------------------------------
# Q8 Reflection:
#
# Both agents received the SAME TOOLS list, including the new
# compute_correlation tool.
#
# The ToolCallingAgent can use the available tools directly.
# It can use load_csv and plot_data, but the plot_data tool
# does not provide a color argument. Therefore, it cannot
# directly add the requested green color through that tool.
#
# The CodeAgent can use the available tools and can also write
# and run Python code. Therefore, it can create the scatter
# plot with matplotlib and explicitly set the dots to green.
#
# This shows that ToolCallingAgent is useful when the required
# operation is already available as a defined tool, while
# CodeAgent is more flexible when the request requires custom
# Python code or a customization that the existing tool does
# not support.
#
# The important difference in this example is the color:
# the ToolCallingAgent's plot_data tool does not support a
# color parameter, while the CodeAgent can generate code that
# changes the scatter-plot color to green.
# ------------------------------------------------------------


# ============================================================
# Q9: Final Reflection
# ============================================================

print("\n--------- Q9 -----------------------------------------------")


# ------------------------------------------------------------
# Reflection:
#
# 1. What is the main difference between ToolCallingAgent
#    and CodeAgent?
#
#    ToolCallingAgent works by selecting and calling the
#    tools that have already been provided. It is a good choice
#    when the required operation already exists as a tool.
#
#    CodeAgent can also use tools, but it can write and execute
#    Python code. This gives it more flexibility for custom
#    analysis, calculations, and visualizations.
#
#
# 2. What did the agents produce for the scatter-plot request?
#
#    Both agents had access to the same TOOLS list. The
#    ToolCallingAgent could use the existing plot_data tool,
#    but that tool does not have a color option. The CodeAgent
#    could generate custom matplotlib code and therefore could
#    make the requested scatter plot with green dots.
#
#
# 3. What does the color comparison show about when each agent
#    is useful?
#
#    The color comparison shows that a ToolCallingAgent is
#    useful for predictable tasks that match the capabilities
#    of existing tools. A CodeAgent is more useful when a task
#    needs a customization that is not supported by the
#    existing tools, such as changing the scatter-plot color.
#
#
# 4. When would you choose each agent?
#
#    I would choose ToolCallingAgent when I have well-defined
#    tools and want the agent to use those tools in a controlled
#    and predictable way.
#
#    I would choose CodeAgent when the task requires flexible
#    Python code, custom calculations, or custom visualizations
#    that cannot be completed by the existing tools alone.
# ------------------------------------------------------------
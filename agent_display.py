import json
from typing import Any, Dict, List, Union

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.tree import Tree
from rich.json import JSON
from rich import box

# Initialize Console
console = Console(width=96)

def print_header(text: str, style: str = "bold cyan") -> None:
    """Prints a styled header."""
    console.print()
    console.print(Panel(text, style=style, box=box.HEAVY, expand=False))

def recursive_json_clean(data: Any) -> Any:
    """
    Recursively traverses a data structure (dict/list) and attempts to parse 
    any string values that look like JSON into actual objects.
    
    This fixes the issue where 'text' fields contain escaped JSON strings 
    like "{\n  \"queryTerm\": ...}" instead of real dicts.
    """
    if isinstance(data, dict):
        return {k: recursive_json_clean(v) for k, v in data.items()}
    
    elif isinstance(data, list):
        return [recursive_json_clean(item) for item in data]
    
    elif isinstance(data, str):
        # Check if the string looks like a JSON object or array
        stripped = data.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or \
           (stripped.startswith("[") and stripped.endswith("]")):
            try:
                parsed = json.loads(data)
                # Recurse in case the parsed object also contains JSON strings
                return recursive_json_clean(parsed)
            except (json.JSONDecodeError, TypeError):
                return data
    
    return data

def print_input(input_data: Any) -> None:
    """Prints the user input."""
    content = input_data
    if isinstance(input_data, dict):
        # In v1.0, 'input' is the standard key for most chains
        content = input_data.get("input") or input_data.get("query") or json.dumps(input_data, indent=2)
    
    console.print(Panel(f"[bold]User Input:[/bold]\n{content}", style="green", title="START", title_align="left"))

def _render_json_or_text(data: Any, style: str = "monokai") -> Union[Syntax, str]:
    """Helper to render data as pretty JSON if possible, else string."""
    
    # 1. Clean the data (Un-stringify nested JSON)
    clean_data = recursive_json_clean(data)

    try:
        # 2. If it's a structure, render as highlighted JSON
        if isinstance(clean_data, (dict, list)):
            text = json.dumps(clean_data, indent=2)
            return Syntax(text, "json", theme=style, word_wrap=True)
    except (TypeError, json.JSONDecodeError):
        pass
    
    # 3. Fallback to string
    return str(clean_data)

def print_intermediate_steps(intermediate_steps: List[Any]) -> None:
    """
    Handles 'intermediate_steps' (Legacy AgentExecutor).
    """
    if not intermediate_steps:
        return

    console.print("\n[bold yellow]🛠️  Tool Execution History (Executor):[/bold yellow]")
    
    for i, step in enumerate(intermediate_steps, 1):
        action, observation = step
        
        tool_name = getattr(action, 'tool', 'Unknown Tool')
        step_tree = Tree(f"[bold]Step {i}: {tool_name}[/bold]")
        
        # Tool Input
        tool_input = getattr(action, 'tool_input', '')
        step_tree.add("[cyan]Input:[/cyan]").add(_render_json_or_text(tool_input))

        # Log
        log_content = getattr(action, 'log', '')
        if log_content:
            clean_log = log_content.split('\n')[0] if '\n' in log_content else log_content
            if tool_name not in clean_log: 
                step_tree.add(f"[italic dim]Log: {clean_log}[/italic dim]")

        # Observation
        step_tree.add("[green]Observation:[/green]").add(_render_json_or_text(observation))
        console.print(step_tree)
        console.print("")

def print_message_history(messages: List[Any]) -> None:
    """
    Handles 'messages' list (LangGraph / v1.0 Standard).
    """
    tool_sequence = []
    
    for msg in messages:
        # v1.0 Standard: AIMessage has .tool_calls
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for call in msg.tool_calls:
                tool_sequence.append({
                    "type": "call",
                    "name": call.get("name"),
                    "args": call.get("args"),
                    "id": call.get("id")
                })
        # v1.0 Standard: ToolMessage represents the result
        elif msg.type == "tool":
            tool_sequence.append({
                "type": "result",
                "content": msg.content,
                "id": msg.tool_call_id,
                "name": msg.name
            })

    if not tool_sequence:
        return

    console.print("\n[bold yellow]🔗 Message History (v1.0 Standard):[/bold yellow]")
    
    for item in tool_sequence:
        if item["type"] == "call":
            tree = Tree(f"[bold]Invoke: {item['name']}[/bold] [dim]({item['id']})[/dim]")
            # Clean input args too
            tree.add("[cyan]Args:[/cyan]").add(_render_json_or_text(item['args']))
            console.print(tree)
        elif item["type"] == "result":
            tree = Tree(f"[bold]Result[/bold] [dim]({item['id']})[/dim]")
            # This is where your messy JSON was appearing. 
            # _render_json_or_text will now clean it recursively.
            tree.add("[green]Output:[/green]").add(_render_json_or_text(item['content']))
            console.print(tree)
            console.print("")

def print_final_answer(output: Any) -> None:
    """Prints the final result."""
    console.print(Panel("[bold]🤖 Final Answer:[/bold]", style="magenta", box=box.MINIMAL))
    
    content = output
    if hasattr(output, 'content'):
        content = output.content

    if isinstance(content, str):
        console.print(Markdown(content))
    elif isinstance(content, dict):
        console.print(JSON.from_data(content))
    else:
        console.print(str(content))
    
    console.print(Panel("End of Interaction", style="dim", title="END", title_align="right"))

def visualize_agent_response(response: Dict[str, Any], show_raw: bool = False) -> None:
    """
    Main Entry Point. Works with LangChain v1.0+
    """
    print_header("LangChain v1.0 Agent Report")

    # 1. Input
    if "input" in response:
        print_input(response["input"])
    elif "messages" in response and len(response["messages"]) > 0:
        print_input(response["messages"][0].content)

    # 2. Intermediate Work
    if "intermediate_steps" in response:
        print_intermediate_steps(response["intermediate_steps"])
    elif "messages" in response:
        print_message_history(response["messages"])
    else:
        console.print("[dim]No tool execution history found.[/dim]\n")

    # 3. Output
    if "output" in response:
        print_final_answer(response["output"])
    elif "messages" in response and len(response["messages"]) > 0:
        print_final_answer(response["messages"][-1])
    
    # 4. Debug
    if show_raw:
        console.print("\n[bold red]🔍 Raw Response Structure:[/bold red]")
        from rich.pretty import Pretty
        console.print(Pretty(response, overflow="fold"))
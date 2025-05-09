import json
import sys
import readline  # Standard library imports
import click
import os
import openai
import time
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.rule import Rule 
from rich.panel import Panel

openai.api_key = "EMPTY"  # no key needed since we use local candle-vllm service
openai.base_url = "http://localhost:2000/v1/"

def clear_console():
    """Clears the console."""
    command = 'cls' if os.name in ('nt', 'dos') else 'clear'
    os.system(command)
    print("\n")

@click.command()
@click.argument("system_prompt", type=str, required=False)
@click.option("--stream", is_flag=True, default=False,
              help="Enable streaming output for responses.")
@click.option("--live", is_flag=True, default=False,
              help="Enable Live update feature (flick in some console).")
@click.option("--max_tokens", type=int, default=1024*16,
              help="Maximum generated tokens for each response.")
@click.option("--frequency", type=int, default=10,
              help="Times per second for output refresh.")
def chatloop(system_prompt: str, stream: bool, live: bool, max_tokens: int, frequency: int):
    """
    A command-line chatbot interface using OpenAI API and candle-vllm as backend.
    """
    console = Console()
    messages = []
    console.clear()

    # Add a system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Main loop
    while True:
        try:
            # User input
            user_input = input("🙋 Please Input: ")
            if user_input == "":
                console.print("Multiline input: press Ctrl+D to finish, Ctrl+C to exit.")
                user_input = sys.stdin.read()
                console.print()
            user_msg = {"role": "user", "content": user_input}

            # Model response
            try:
                with Live(Spinner("dots", text="Connecting...", style="green"), transient=True, console=console):
                    response = openai.chat.completions.create(
                        model="Anything you want!",
                        messages=messages + [user_msg],
                        stream=True,
                        max_tokens = max_tokens,
                    )
                
                console.print(Rule(title="Candle-vLLM:", align="left", style="cyan"))
                # Handle streaming response
                msg = ""
                if live:
                    with Live(Panel("", title="Candle-vLLM Response", border_style="cyan"),
                              console=console,
                              auto_refresh=True,
                              refresh_per_second=frequency,
                              vertical_overflow="visible") as l:
                        for chunk in response:
                            content = chunk.choices[0].delta.content
                            if content:
                                msg += content
                                l.update(Panel(Markdown(msg), title="Candle-vLLM Response", border_style="cyan"))
  
                else:
                    for chunk in response:
                        content = chunk.choices[0].delta.content
                        if content:
                            msg += content
                            console.out(content, end="")
                    console.out("")
                    console.print(Rule(style="cyan"), "")

                # Save conversation history
                messages.append(user_msg)
                messages.append({"role": "assistant", "content": msg})

                if live:
                    clear_console() # clear repetitive live outputs
                    #reprint all conversation messages
                    console.clear()
                    for m in messages:
                        if m["role"] == "user":
                           console.out(" 🙋 Please Input: ", end="")
                           console.out(m["content"])
                        else:
                            with Live(Panel("", title="Candle-vLLM Response", border_style="cyan"),
                                    console=console,
                                    auto_refresh=True,
                                    refresh_per_second=frequency,
                                    vertical_overflow="visible") as l:
                                l.update(Panel(Markdown(m["content"]), title="Candle-vLLM Response", border_style="cyan"))
                    console.out("")
                    console.print(Rule(style="cyan"), "")
            except KeyboardInterrupt:
                console.log("Response interrupted by user. Press Ctrl+C again to exit.", style="yellow")
                continue
            except openai.error.AuthenticationError as e:
                console.log(f"Authentication error: {e}", style="bold red")
                break
            except openai.error.OpenAIError as e:
                console.log(f"Unexpected OpenAI error: {e}", style="bold red")
        
        except KeyboardInterrupt:
            console.print("\nExiting.", style="bold green")
            break

if __name__ == "__main__":
    chatloop()

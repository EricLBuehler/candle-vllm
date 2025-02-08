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
openai.api_key = "EMPTY" # no key needed since we use local candle-vllm service
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
@click.option("--max_tokens", type=int, default=1024*16,
              help="Maximum generated tokens for each response.")
@click.option("--frequency", type=int, default=10,
              help="Times per second for output refresh.")
def chatloop(system_prompt: str, stream: bool, max_tokens: int, frequency: int):
    """
    A command-line chatbot interface using OpenAI API and candle-vllm as backend.
    
    :param system_prompt: Initial prompt for the chatbot.
    :param stream: Flag to enable streaming responses.
    """
    console = Console()
    messages = []
    clear_console()
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
                with Live(console=console, auto_refresh=True, refresh_per_second=frequency, vertical_overflow="visible") as live:
                    for chunk in response:
                        content = chunk.choices[0].delta.content
                        if content != None:
                            msg += content
                            live.update(msg)
                
                clear_console() # clear repetitive live outputs
                console.print(msg) #show final full results
                console.print(Rule(style="cyan"), "")
                # Save conversation history
                messages.append(user_msg)
                messages.append({"role": "assistant", "content": msg})
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

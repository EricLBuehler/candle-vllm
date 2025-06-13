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
from typing import Optional
openai.api_key = "EMPTY"  # no key needed since we use local candle-vllm service
openai.base_url = "http://localhost:2000/v1/"

def clear_console():
    """Clears the console."""
    command = 'cls' if os.name in ('nt', 'dos') else 'clear'
    os.system(command)
    print("\n")

@click.command()
@click.option("--system_prompt", type=str, default=None, 
              help="System prompt for model request")
@click.option("--stream", is_flag=True, default=False,
              help="Enable streaming output for responses.")
@click.option("--live", is_flag=True, default=False,
              help="Enable Live update feature (flick in some console).")
@click.option("--max_tokens", type=int, default=1024*16,
              help="Maximum generated tokens for each response.")
@click.option("--frequency", type=int, default=10,
              help="Times per second for output refresh.")
@click.option("--port", type=int, default=2000,
              help="Server port.")
@click.option("--temperature", type=float, default=None,
            help="Sampling temperature")
@click.option("--top_p", type=float, default=None,
            help="Sampling top-p")
@click.option("--top_k", type=int, default=None,
            help="Sampling top-k")
@click.option("--thinking", type=bool, default=None,
              help="Enable thinking for reasoning models.")
def chatloop(system_prompt: Optional[str], stream: bool, live: bool, 
    max_tokens: int, frequency: int, port: int, temperature: Optional[float], top_k: Optional[int], top_p: Optional[float], thinking: Optional[bool]):
    """
    A command-line chatbot interface using OpenAI API and candle-vllm as backend.
    """
    console = Console()
    messages = []
    console.clear()
    openai.base_url = "http://localhost:"+str(port)+"/v1/"

    # Add a system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Main loop
    while True:
        try:
            # User input
            user_input = input("🙋 Please Input (Ctrl+C to start a new chat or exit): ")
            if user_input == "":
                console.print("Multiline input: press Ctrl+D to finish, Ctrl+C to exit.")
                user_input = sys.stdin.read()
                console.print()
            user_msg = {"role": "user", "content": user_input}
            extra_body = {"top_k": top_k, "thinking": thinking}
            # Model response
            try:
                with Live(Spinner("dots", text="Connecting...", style="green"), transient=True, console=console):
                    response = openai.chat.completions.create(
                        model="Anything you want!",
                        messages=messages + [user_msg],
                        stream=True,
                        max_tokens = max_tokens,
                        temperature = temperature,
                        top_p = top_p,
                        extra_body = extra_body,
                    )
                
                console.print(Rule(title="Candle-vLLM:", align="left", style="cyan"))
                # Handle streaming response
                prefix = ""
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
                                if msg == "" and prefix == "" and content[0] == "<":
                                    prefix = content # <think> tag can cause Markdown problem in the first line
                                msg += content
                                l.update(Panel(Markdown(prefix + msg), title="Candle-vLLM Response", border_style="cyan"))
  
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
                                l.update(Panel(Markdown(prefix + m["content"]), title="Candle-vLLM Response", border_style="cyan"))
                    console.out("")
                    console.print(Rule(style="cyan"), "")
            except KeyboardInterrupt:
                console.log("Response interrupted by user. Press Ctrl+C again to exit.", style="yellow")
                continue
            except Exception as e:
                console.log(f"Request error: {e}", style="bold red")
                break
            except openai.error.OpenAIError as e:
                console.log(f"Unexpected OpenAI error: {e}", style="bold red")
        
        except KeyboardInterrupt:
            if len(messages) == 0 or (len(messages) == 1 and messages[0]["role"]=="system"):
                console.print("\nExiting.", style="bold green")
                break

            messages.clear()
            # Reinsert the system prompt for the next chat session
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            console.clear()
            console.log("A new chat is started. Press Ctrl+C again to exit.", style="yellow")
            continue
 

if __name__ == "__main__":
    chatloop()

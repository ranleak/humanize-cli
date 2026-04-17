import os
import re
import sys
from typing import Tuple, List

# Ensure the user has the required libraries
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
    from rich.rule import Rule
    from openai import OpenAI
except ImportError:
    print("Please install the required libraries:")
    print("pip install rich openai")
    sys.exit(1)

# Initialize Rich Console
console = Console()

# Define common AI tropes and their algorithmic "human" replacements
AI_TROPES_MAP = {
    r"\bdelve into\b": "explore",
    r"\bin conclusion\b": "to wrap up",
    r"\ba testament to\b": "proof of",
    r"\btapestry\b": "mix",
    r"\bplethora\b": "lot",
    r"\bit is important to note that\b": "keep in mind that",
    r"\bmultifaceted\b": "complex",
    r"\bmoreover\b": "plus",
    r"\bnavigating the landscape\b": "figuring out",
    r"\brealm\b": "world",
    r"\bembark on a journey\b": "start",
    r"\bunderscore\b": "highlight",
    r"\bsynergy\b": "teamwork",
    r"\bparadigm shift\b": "big change",
    r"\bcatalyst\b": "spark",
    r"\bseamlessly\b": "smoothly",
}

def check_api_key() -> str:
    """Check if the Vercel AI Gateway API key is set."""
    api_key = os.getenv("AI_GATEWAY_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/] AI_GATEWAY_API_KEY environment variable is missing.")
        console.print("Please set it using: [bold cyan]export AI_GATEWAY_API_KEY='your_key_here'[/]")
        sys.exit(1)
    return api_key

def algorithmic_preprocess(text: str) -> Tuple[str, List[str]]:
    """
    Finds AI tropes, highlights them, and applies a basic algorithmic 
    substitution to give the LLM a better starting point.
    """
    found_tropes = []
    preprocessed_text = text

    for trope_pattern, replacement in AI_TROPES_MAP.items():
        # Case-insensitive search
        matches = re.findall(trope_pattern, preprocessed_text, flags=re.IGNORECASE)
        if matches:
            found_tropes.extend(matches)
            # Apply algorithmic substitution
            preprocessed_text = re.sub(
                trope_pattern, 
                replacement, 
                preprocessed_text, 
                flags=re.IGNORECASE
            )

    return preprocessed_text, list(set(found_tropes))

def highlight_text(original_text: str, tropes: List[str]) -> Text:
    """Highlights found tropes in the original text using Rich."""
    styled_text = Text(original_text)
    for trope in tropes:
        styled_text.highlight_words([trope], "bold red on yellow")
    return styled_text

def call_llm_humanizer(client: OpenAI, preprocessed_text: str, tropes: List[str]) -> str:
    """Calls the Vercel AI Gateway using Grok to fully humanize the text."""
    
    # We pass the found tropes so the LLM explicitly knows what to avoid
    trope_warning = (
        f"I've already removed these AI-like words: {', '.join(tropes)}. " if tropes else ""
    )

    system_prompt = (
        "You are an expert human editor. Your job is to take text and make it sound completely "
        "natural, conversational, and written by a real human being. Avoid all common AI tropes "
        "(like 'delve', 'tapestry', overly balanced sentence structures, and robotic transitions). "
        "Write with voice, slight imperfections, and natural cadence. Do not add conversational filler "
        "like 'Sure, here is your text'. Just return the final humanized text."
    )

    user_prompt = (
        f"Please humanize the following text. {trope_warning}\n\n"
        f"Text to humanize:\n{preprocessed_text}"
    )

    try:
        response = client.chat.completions.create(
            model='xai/grok-4.1-fast-reasoning',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            temperature=0.7, # A bit of randomness helps with natural human cadence
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[red]An error occurred while calling the AI Gateway:[/] {e}"

def main():
    console.clear()
    console.print(Panel.fit("[bold cyan]🤖 AI to Human Text Converter[/bold cyan]\nPowered by Vercel AI Gateway & xAI Grok", border_style="cyan"))
    
    # Initialize OpenAI Client via Vercel AI Gateway
    api_key = check_api_key()
    client = OpenAI(
        api_key=api_key,
        base_url='https://ai-gateway.vercel.sh/v1'
    )

    while True:
        console.print("\n")
        original_text = Prompt.ask("[bold green]Paste the AI text you want to humanize[/] (or type 'quit' to exit)")
        
        if original_text.lower() in ['quit', 'exit', 'q']:
            console.print("[bold yellow]Goodbye![/]")
            break
            
        if not original_text.strip():
            continue

        # Step 1: Algorithmic Preprocessing
        console.print(Rule("Step 1: Algorithmic Analysis", style="blue"))
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Scanning for AI tropes...", total=None)
            preprocessed_text, found_tropes = algorithmic_preprocess(original_text)
            
        if found_tropes:
            console.print(f"[bold red]Found {len(found_tropes)} AI trope(s):[/] {', '.join(found_tropes)}")
            console.print("\n[bold]Original Text with Tropes Highlighted:[/bold]")
            console.print(Panel(highlight_text(original_text, found_tropes), border_style="yellow"))
        else:
            console.print("[green]No obvious AI tropes found in the algorithmic pass.[/green]")

        # Step 2: LLM Deep Humanization
        console.print(Rule("Step 2: LLM Deep Humanization", style="magenta"))
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Rewriting with xAI Grok (grok-4.1-fast-reasoning)...", total=None)
            final_humanized_text = call_llm_humanizer(client, preprocessed_text, found_tropes)

        console.print("\n[bold]Final Humanized Text:[/bold]")
        console.print(Panel(Markdown(final_humanized_text), border_style="green"))

if __name__ == "__main__":
    main()
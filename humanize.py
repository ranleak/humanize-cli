"""
AI Text Humanizer Pipeline
Implements algorithmic humanization to naturalize AI-generated text.

Prerequisites:
    pip install rich spacy nltk transformers torch numpy
    python -m spacy download en_core_web_sm
"""

import time
import random
import numpy as np
from typing import List, Tuple

# Rich UI Imports
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich import print as rprint

# NLP Imports
import spacy
import nltk
from nltk.corpus import wordnet
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel

# Initialize Rich Console
console = Console()

class AITextHumanizer:
    def __init__(self):
        self.nlp = None
        self.para_tokenizer = None
        self.para_model = None
        self.embed_tokenizer = None
        self.embed_model = None
        
    def load_models(self, progress: Progress, task_id: int):
        """Loads all required NLP models from spaCy, NLTK, and HF Transformers."""
        # 1. Load spaCy
        progress.update(task_id, description="[cyan]Loading spaCy (en_core_web_sm)...")
        self.nlp = spacy.load("en_core_web_sm")
        
        # 2. Ensure NLTK resources
        progress.update(task_id, description="[cyan]Loading NLTK WordNet...")
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        # 3. Load Hugging Face Paraphraser (T5-small for speed/efficiency)
        progress.update(task_id, description="[cyan]Loading HF Transformers (Paraphraser)...")
        self.para_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.para_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        
        # 4. Load Hugging Face Embedder for Coherence Check
        progress.update(task_id, description="[cyan]Loading HF Transformers (Embeddings)...")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embed_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embed_model = AutoModel.from_pretrained(model_name)
        
        progress.update(task_id, completed=100, description="[green]All models loaded successfully!")

    def get_wordnet_pos(self, spacy_pos: str) -> str:
        """Map spaCy POS tags to WordNet POS tags."""
        if spacy_pos.startswith('J'):
            return wordnet.ADJ
        elif spacy_pos.startswith('V'):
            return wordnet.VERB
        elif spacy_pos.startswith('N'):
            return wordnet.NOUN
        elif spacy_pos.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def step_lexical_variation(self, text: str) -> str:
        """Step 3: Strategic synonym replacement targeting verbs & adjectives."""
        doc = self.nlp(text)
        new_tokens = []
        
        for token in doc:
            # Target ~25% of Adjectives and Verbs for substitution
            if token.pos_ in ["ADJ", "VERB"] and random.random() < 0.25:
                wn_pos = self.get_wordnet_pos(token.tag_)
                synonyms = []
                if wn_pos:
                    for syn in wordnet.synsets(token.text, pos=wn_pos):
                        for lemma in syn.lemmas():
                            lemma_name = lemma.name().replace('_', ' ')
                            # Keep it structurally simple and different from original
                            if lemma_name.lower() != token.text.lower():
                                synonyms.append(lemma_name)
                
                if synonyms:
                    # Choose a synonym and try to match original casing
                    choice = random.choice(synonyms)
                    if token.is_title:
                        choice = choice.title()
                    elif token.is_upper:
                        choice = choice.upper()
                    new_tokens.append(choice + token.whitespace_)
                else:
                    new_tokens.append(token.text_with_ws)
            else:
                new_tokens.append(token.text_with_ws)
                
        return "".join(new_tokens)

    def step_structure_diversify(self, text: str) -> str:
        """Step 4: Increasing burstiness via Hugging Face T5 paraphrasing."""
        doc = self.nlp(text)
        diversified_sentences = []
        
        for sent in doc.sents:
            # ~30% chance to deeply paraphrase a sentence to alter length/structure
            if random.random() < 0.30:
                input_text = f"paraphrase: {sent.text}"
                input_ids = self.para_tokenizer(input_text, return_tensors="pt").input_ids
                
                with torch.no_grad():
                    outputs = self.para_model.generate(
                        input_ids, 
                        max_length=60, 
                        num_beams=4, 
                        early_stopping=True
                    )
                
                para_text = self.para_tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Fallback if generation failed or is too short
                if len(para_text) > 5: 
                    diversified_sentences.append(para_text)
                else:
                    diversified_sentences.append(sent.text)
            else:
                diversified_sentences.append(sent.text)
                
        return " ".join(diversified_sentences)

    def step_inject_natural_elements(self, text: str) -> str:
        """Step 5: Inject contextual imperfections and human conversational elements."""
        doc = self.nlp(text)
        
        fillers = ["Actually,", "You know,", "To be honest,", "Look —", "So,", "Now,", "In fact,"]
        contractions = {
            r"\bit is\b": "it's", r"\bIt is\b": "It's",
            r"\bdo not\b": "don't", r"\bDo not\b": "Don't",
            r"\bcannot\b": "can't", r"\bCannot\b": "Can't",
            r"\bwill not\b": "won't", r"\bWill not\b": "Won't",
            r"\bhave not\b": "haven't", r"\bI am\b": "I'm",
        }
        
        import re
        current_text = text
        # Apply Contractions
        for pattern, replacement in contractions.items():
            current_text = re.sub(pattern, replacement, current_text)
            
        # Apply Fillers
        doc_updated = self.nlp(current_text)
        new_sents = []
        for i, sent in enumerate(doc_updated.sents):
            sent_str = sent.text.strip()
            # 20% chance to add a conversational filler to sentences (except the very first one usually)
            if i > 0 and random.random() < 0.2:
                sent_str = f"{random.choice(fillers)} {sent_str[:1].lower()}{sent_str[1:]}"
            new_sents.append(sent_str)
            
        return " ".join(new_sents)

    def get_embedding(self, text: str) -> np.ndarray:
        """Generates mean-pooled sentence embeddings for coherence checking."""
        inputs = self.embed_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
            
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return (sum_embeddings / sum_mask)[0].numpy()

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculates cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def process(self, original_text: str) -> Tuple[str, float]:
        """Runs the full coordinated transformation pipeline."""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("[cyan]1 & 2. Tokenizing & POS Tagging (spaCy)...", total=100)
            self.nlp(original_text) # Warmup
            time.sleep(0.5)
            progress.update(task, completed=100)
            
            task2 = progress.add_task("[magenta]3. Lexical Variation (NLTK WordNet)...", total=100)
            stage1_text = self.step_lexical_variation(original_text)
            time.sleep(0.8)
            progress.update(task2, completed=100)
            
            task3 = progress.add_task("[blue]4. Diversifying Structure (HF T5)...", total=100)
            stage2_text = self.step_structure_diversify(stage1_text)
            time.sleep(1.2)
            progress.update(task3, completed=100)
            
            task4 = progress.add_task("[yellow]5. Injecting Natural Elements...", total=100)
            final_draft = self.step_inject_natural_elements(stage2_text)
            time.sleep(0.6)
            progress.update(task4, completed=100)
            
            task5 = progress.add_task("[green]6. Semantic Coherence Check (MiniLM)...", total=100)
            orig_emb = self.get_embedding(original_text)
            final_emb = self.get_embedding(final_draft)
            sim_score = self.cosine_similarity(orig_emb, final_emb)
            time.sleep(0.5)
            progress.update(task5, completed=100)
            
            # Rollback logic per the architecture spec
            if sim_score < 0.85:
                console.print(f"[bold red]! Semantic drift detected (Similarity: {sim_score:.2f}). Rolling back to Stage 1.[/bold red]")
                final_draft = stage1_text
                sim_score = self.cosine_similarity(orig_emb, self.get_embedding(final_draft))
                
        return final_draft.strip(), sim_score

def main():
    console.print(Panel.fit(
        "[bold cyan]AI Text Humanization Engine[/bold cyan]\n"
        "Implementing Perplexity, Burstiness, & Lexical Diversity adjustments.",
        border_style="cyan"
    ))
    
    humanizer = AITextHumanizer()
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("[cyan]Initializing Engine...", total=100)
        humanizer.load_models(progress, task)

    while True:
        console.print("\n[bold]Enter AI-generated text to humanize[/bold] (or type 'quit' to exit):")
        original_text = Prompt.ask(">>")
        
        if original_text.lower() in ['quit', 'exit', 'q']:
            break
            
        if not original_text.strip():
            continue
            
        console.print("\n[bold]Initiating Processing Pipeline...[/bold]")
        humanized_text, score = humanizer.process(original_text)
        
        # Display Results
        table = Table(title="Humanization Results", show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Original (Machine-Generated)", style="dim", width=45)
        table.add_column("Humanized Output", style="green", width=45)
        
        table.add_row(original_text, humanized_text)
        
        console.print("\n")
        console.print(table)
        
        # Coherence feedback
        score_color = "green" if score >= 0.85 else "red"
        console.print(f"[*] Semantic Coherence (Cosine Similarity): [bold {score_color}]{score:.3f}[/bold {score_color}] "
                      f"(Target: > 0.85)")

if __name__ == "__main__":
    main()
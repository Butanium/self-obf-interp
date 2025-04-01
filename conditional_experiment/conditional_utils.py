from collections import defaultdict
import pickle
from jaxtyping import Int64
from obf_reps.logging import Logger
from obf_reps.metrics import TrainableMetric
from obf_reps.models.hf import HFHardPrompted, HFHardPromptedWithSelectableLayers
from obf_reps.models import ModelConfig
from pathlib import Path
from torch import Tensor
from typing import Dict, List, Optional, Tuple
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import torch


def load_word_to_probe_dict(probe_checkpoint_dir: str) -> dict[str, TrainableMetric]:
    """
    Load a dictionary mapping words to their corresponding probe objects from a given directory.

    The function assumes that each probe is saved as a .pkl file in the specified directory,
    where the filename (without the .pkl extension) is the word. For example, a file named
    "bomb.pkl" will be loaded as the probe for the word "bomb".

    Args:
        probe_checkpoint_dir (str): Path to the directory containing the probe checkpoint files.

    Returns:
        dict[str, TrainableMetric]: A dictionary mapping word (str) to the loaded probe object.
    """
    if not os.path.isdir(probe_checkpoint_dir):
        raise ValueError(f"Provided path is not a directory: {probe_checkpoint_dir}")

    word_to_probe: dict[str, TrainableMetric] = {}
    # Iterate over all files in the directory
    for filename in os.listdir(probe_checkpoint_dir):
        if filename.endswith(".pkl"):
            word = filename[:-4]  # Remove the ".pkl" extension to get the word
            word = word.replace("train_", "").replace("test_", "")
            file_path = os.path.join(probe_checkpoint_dir, filename)
            with open(file_path, "rb") as f:
                probe = pickle.load(f)
            if not isinstance(probe, TrainableMetric):
                raise ValueError(f"Loaded object is not a TrainableMetric: {file_path}")
            word_to_probe[word] = probe

    return word_to_probe


def load_local_model(
    checkpoint_path: Optional[str] = None, model_name: str = "llama_3_8b_instruct"
):
    """Load the Llama model, optionally from a checkpoint.

    Args:
        checkpoint_path: Optional path to a model checkpoint. If None, loads base model.
    """
    print(f"Loading model...")
    base_model_path = Path(f"/workspace/{model_name}")
    model_config = ModelConfig()

    if checkpoint_path is not None:
        # Load model weights from checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map=model_config.device,
            torch_dtype=model_config.model_dtype,
        )
        model.eval()
        if not model_config.requires_grad:
            for p in model.parameters():
                p.requires_grad = False

        # Load and configure tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token:
            pass
        elif tokenizer.unk_token:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eos_token:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        model.generation_config.pad_token_id = tokenizer.pad_token_id

        # Create the wrapped model
        model = HFHardPromptedWithSelectableLayers(
            model=model, tokenizer=tokenizer, config=model_config
        )
    else:
        # Load everything from base model
        model = HFHardPromptedWithSelectableLayers.load_model(
            base_model_path, model_config
        )

    return model


def sanitize_response(response: str) -> str:
    specials = ["<|eot_id|>", "<pad>", "<end_of_turn>", "<eos>"]
    for special in specials:
        response = response.replace(special, "")
    return response.strip()


def get_llm_responses(
    model: HFHardPromptedWithSelectableLayers,
    prompts: List[str],
    max_tokens: int = 100,
    temperature: float = 0.7,
    deterministic: bool = False,
    ensure_different_responses: bool = True,
    max_retries: int = 3,
    logger: Logger = None,
) -> Tuple[List[str], List[Int64[Tensor, "gen_len"]]]:
    """Get a response from local model for the given prompt(s). Optionally ensure responses are unique.

    Returns:
        Tuple containing:
        - List of response strings
        - List of token IDs for each response
    """
    logger.print(f"Getting responses for {len(prompts)} prompts...")
    logger.print(
        f"Parameters: max_tokens={max_tokens}, temperature={temperature}, deterministic={deterministic}"
    )

    if not ensure_different_responses:
        logger.optional_print(
            "Not ensuring different responses - using original behavior"
        )
        # Original behavior
        if len(prompts) > 800:
            # Split into batches of 800
            all_responses = []
            all_token_ids = []
            for i in range(0, len(prompts), 800):
                logger.print(
                    f"Generating responses for {len(prompts[i:i+800])} prompts (batch {i//800+1}/{len(prompts)//800})..."
                )
                batch = prompts[i : i + 800]
                gen_return = model.generate_from_string(
                    input_text=batch,
                    use_tunable_params=False,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=not deterministic,
                    add_chat_template=True,
                )
                all_responses.extend(gen_return.gen_text)
                all_token_ids.extend([ids for ids in gen_return.gen_ids])
            return all_responses, all_token_ids
        else:
            gen_return = model.generate_from_string(
                input_text=prompts,
                use_tunable_params=False,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=not deterministic,
                add_chat_template=True,
            )
            return gen_return.gen_text, [ids for ids in gen_return.gen_ids]

    logger.optional_print("Ensuring different responses with retry logic")
    # New behavior to ensure different responses with batched generation
    final_responses = [""] * len(prompts)  # Pre-allocate list
    final_token_ids = [None] * len(prompts)  # Pre-allocate list for token IDs
    remaining_indices = list(range(len(prompts)))
    curr_temp = temperature
    attempts = 0

    inference_batch_size = 500
    while remaining_indices and attempts < max_retries:
        logger.optional_print(
            f"\nAttempt {attempts+1}/{max_retries} with temperature {curr_temp}"
        )
        logger.optional_print(
            f"Generating responses for {len(remaining_indices)} remaining prompts..."
        )

        # Batch generate responses for all remaining prompts
        remaining_prompts = [prompts[i] for i in remaining_indices]
        if (
            len(remaining_prompts) > inference_batch_size
        ):  # if running out of memory, lower batch size
            # Split into batches of inference_batch_size
            batch_responses = []
            batch_token_ids = []
            for i in range(0, len(remaining_prompts), inference_batch_size):
                logger.print(
                    f"Generating responses for {len(remaining_prompts[i:i+inference_batch_size])} prompts (batch {i//inference_batch_size+1}/{len(remaining_prompts)//inference_batch_size})..."
                )
                batch = remaining_prompts[i : i + inference_batch_size]
                gen_return = model.generate_from_string(
                    input_text=batch,
                    use_tunable_params=False,
                    max_new_tokens=max_tokens,
                    temperature=curr_temp,
                    do_sample=not deterministic,
                    add_chat_template=True,
                )
                batch_responses.extend(gen_return.gen_text)
                batch_token_ids.extend([ids for ids in gen_return.gen_ids])
            gen_return.gen_text = batch_responses
            gen_ids = batch_token_ids
        else:
            gen_return = model.generate_from_string(
                input_text=remaining_prompts,
                use_tunable_params=False,
                max_new_tokens=max_tokens,
                temperature=curr_temp,
                do_sample=not deterministic,
                add_chat_template=True,
            )
            gen_ids = [ids for ids in gen_return.gen_ids]

        # Process responses and track which ones need to be regenerated
        still_remaining = []
        for idx_pos, idx in enumerate(remaining_indices):
            response = sanitize_response(gen_return.gen_text[idx_pos])
            logger.optional_print(f"Generated response for prompt {idx+1}: {response}")

            # Check if this response is already used for any prompt
            if response in final_responses:
                logger.optional_print(
                    f"Response for prompt {idx+1} was duplicate - will retry"
                )
                still_remaining.append(idx)
            else:
                logger.optional_print(f"Response for prompt {idx+1} accepted")
                final_responses[idx] = response
                final_token_ids[idx] = gen_ids[idx_pos]

        remaining_indices = still_remaining
        attempts += 1
        # Increase temperature for next attempt
        curr_temp = temperature * (1.5**attempts)

        if remaining_indices:
            logger.optional_print(
                f"\nStill need responses for prompts: {[i+1 for i in remaining_indices]}"
            )
            logger.optional_print(f"Retrying with temperature increased to {curr_temp}")

    # Fill in any remaining responses that we couldn't make unique
    if remaining_indices:
        logger.print(
            f"\nWarning: Could not generate unique responses for all prompts after {max_retries} attempts"
        )
        logger.print(
            f"Using last generated responses for {len(remaining_indices)} prompts: {[i+1 for i in remaining_indices]}"
        )
        for pos, idx in enumerate(remaining_indices):
            final_responses[idx] = sanitize_response(gen_return.gen_text[pos])
            final_token_ids[idx] = gen_ids[pos]

    logger.print(f"\nFinished generating {len(final_responses)} responses")
    return final_responses, final_token_ids


class WordToPromptFormatter:
    def __init__(self):
        self.formatter_name = "Unimplemented"

    def reset(self):
        pass

    def get_prompt(self, word: str) -> str:
        pass


class PleaseWritePromptFormatter(WordToPromptFormatter):
    def __init__(self):
        self.formatter_name = "PleaseWritePrompt"

    def get_prompt(self, word: str) -> str:
        return f"Please write a sentence that includes the word '{word}'."


class CutoutVariedPromptFormatter(WordToPromptFormatter):
    def __init__(self, seed: int = 42, word_removal_chance: float = 0.5):
        self.formatter_name = "CutoutVariedPrompt"
        self.varied_formatter = VariedDirectPromptFormatter(seed)

    def get_prompt(self, word: str) -> str:
        prompt = self.varied_formatter.get_prompt(word)
        if random.random() < self.word_removal_chance:
            # Remove word with and without surrounding spaces/quotes
            replacements = [
                f" {word}",
                f"{word} ",
                f" '{word}'",
                f"'{word}' ",
                f"'{word}'",
                word,
            ]
            for pattern in replacements:
                prompt = prompt.replace(pattern, "")
        return prompt


class TwoWordVariedDirectPromptFormatter(WordToPromptFormatter):
    def __init__(self, seed: int = 42):
        self.formatter_name = "TwoWordVariedDirectPrompt"
        self.last_taken_index = -1
        self.seed = seed
        random.seed(seed)

        # Create the list first, then shuffle it
        self.shuffled_prompts = [
            # Simple prompts
            lambda word1, word2: f"Write a sentence about '{word1}' and '{word2}'.",
            lambda word1, word2: f"Write a sentence that includes the words '{word1}' and '{word2}'.",
            lambda word1, word2: f"Please write about '{word1}' and '{word2}'.",
            lambda word1, word2: f"Please write a sentence that includes the words '{word1}' and '{word2}'.",
            lambda word1, word2: f"Compose a sentence featuring the words '{word1}' and '{word2}'.",
            lambda word1, word2: f"Create a sentence using '{word1}' and '{word2}' in any context.",
            lambda word1, word2: f"Develop a sentence that incorporates '{word1}' and '{word2}'.",
            lambda word1, word2: f"Craft a sentence demonstrating the use of '{word1}' and '{word2}'.",
            lambda word1, word2: f"Design a sentence where '{word1}' and '{word2}' play a role.",
            lambda word1, word2: f"Express a thought using '{word1}' and '{word2}' in a sentence.",
            lambda word1, word2: f"Produce a sentence with '{word1}' and '{word2}' in it.",
            # Question-based prompts
            lambda word1, word2: f"Can you write a sentence combining {word1} and {word2}?",
            lambda word1, word2: f"Can you create a sentence using the words '{word1}' and '{word2}'?",
            lambda word1, word2: f"Would you write a sentence featuring '{word1}' and '{word2}'?",
            lambda word1, word2: f"How would you use '{word1}' and '{word2}' in a meaningful sentence?",
            lambda word1, word2: f"Can you write about {word1} and {word2}?",
            lambda word1, word2: f"What's a good sentence that includes '{word1}' and '{word2}'?",
            lambda word1, word2: f"Could you demonstrate the use of '{word1}' and '{word2}' in a sentence?",
            lambda word1, word2: f"Tell me how '{word1}' and '{word2}' might be related.",
            lambda word1, word2: f"How would you connect '{word1}' and '{word2}'?",
            # Multi-sentence prompts
            lambda word1, word2: f"Think about the words '{word1}' and '{word2}'. Now create a sentence using both.",
            lambda word1, word2: f"Consider '{word1}' and '{word2}'. Write a sentence that connects them.",
            lambda word1, word2: f"Let's work with '{word1}' and '{word2}'. Please incorporate both into a sentence.",
            lambda word1, word2: f"I'm interested in '{word1}' and '{word2}'. Show me how you'd use them in a sentence.",
            lambda word1, word2: f"Here are your words: '{word1}' and '{word2}'. Create a sentence using both.",
            # Creative/specific direction prompts
            lambda word1, word2: f"Craft a descriptive sentence featuring both '{word1}' and '{word2}'.",
            lambda word1, word2: f"Compose a sentence that connects '{word1}' and '{word2}' effectively.",
            lambda word1, word2: f"Build a sentence around '{word1}' and '{word2}', making them both important.",
            lambda word1, word2: f"Write a sentence where '{word1}' and '{word2}' work together naturally.",
            lambda word1, word2: f"Create a vivid sentence incorporating both '{word1}' and '{word2}'.",
            # Context-providing prompts
            lambda word1, word2: f"Imagine a scenario involving '{word1}' and '{word2}'. Write a sentence about it.",
            lambda word1, word2: f"Picture a situation connecting '{word1}' and '{word2}'. Describe it in one sentence.",
            lambda word1, word2: f"Taking '{word1}' and '{word2}' as inspiration, construct a meaningful sentence.",
            lambda word1, word2: f"Using '{word1}' and '{word2}' as foundations, develop an interesting sentence.",
            lambda word1, word2: f"With '{word1}' and '{word2}' in mind, write a sentence that tells a mini-story.",
            # Style-specific prompts
            lambda word1, word2: f"Write a recipe combining '{word1}' and '{word2}'. Use both words.",
            lambda word1, word2: f"Compose a short story connecting '{word1}' and '{word2}'.",
            lambda word1, word2: f"Please write a poem about '{word1}' and '{word2}'.",
            lambda word1, word2: f"List ways that '{word1}' and '{word2}' might interact.",
            lambda word1, word2: f"Can you write lyrics mentioning both '{word1}' and '{word2}'?",
        ]
        random.shuffle(self.shuffled_prompts)  # Shuffle in place

    def reset(self):
        self.last_taken_index = -1

    def get_prompt(self, word1: str, word2: str) -> str:
        sentence = self.shuffled_prompts[self.last_taken_index](word1, word2)
        self.last_taken_index = (self.last_taken_index + 1) % len(self.shuffled_prompts)
        # Chance of removing or replacing quotations
        if random.random() < 0.33:
            sentence = sentence.replace("'", "")
        elif random.random() < 0.66:
            sentence = sentence.replace("'", "").replace("'", "")

        # Chance of removing punctuation
        if random.random() < 0.5:
            sentence = sentence.replace(".", "").replace("!", "").replace("?", "")
        return sentence


class VariedDirectPromptFormatter(WordToPromptFormatter):
    def __init__(self, seed: int = 42):
        self.formatter_name = "VariedDirectPrompt"
        self.last_taken_index = -1
        self.seed = seed
        random.seed(seed)

        # Create the list first, then shuffle it
        self.shuffled_prompts = [
            # Simple prompts
            lambda word: f"Write a sentence about '{word}'.",
            lambda word: f"Write a sentence that includes the word '{word}'.",
            lambda word: f"Please write about '{word}'.",
            lambda word: f"Please write a sentence that includes the word '{word}'.",
            lambda word: f"Compose a sentence featuring the word '{word}'.",
            lambda word: f"Create a sentence using '{word}' in any context.",
            lambda word: f"Develop a sentence that incorporates '{word}'.",
            lambda word: f"Craft a sentence demonstrating the use of '{word}'.",
            lambda word: f"Design a sentence where '{word}' plays a role.",
            lambda word: f"Express a thought using '{word}' in a sentence.",
            lambda word: f"Produce a sentence with '{word}' in it.",
            # Question-based prompts
            lambda word: f"Can you write a {word}-themed sentence?",
            lambda word: f"Can you create a sentence using the word '{word}'?",
            lambda word: f"Would you write a sentence featuring '{word}'?",
            lambda word: f"How would you use '{word}' in a meaningful sentence?",
            lambda word: f"Can you write a {word}-themed essay?",
            lambda word: f"What's a good sentence that includes '{word}'?",
            lambda word: f"Could you demonstrate the use of '{word}' in a sentence?",
            lambda word: f"Tell me everything you know about '{word}'.",
            lambda word: f"What does '{word}' mean?",
            # Multi-sentence prompts
            lambda word: f"Think about the word '{word}'. Now create a sentence using it.",
            lambda word: f"Consider the meaning of '{word}'. Write a sentence that captures its essence.",
            lambda word: f"Let's work with the word '{word}'. Please incorporate it into a sentence.",
            lambda word: f"I'm interested in the word '{word}'. Show me how you'd use it in a sentence.",
            lambda word: f"Here's your word: '{word}'. Create a sentence that highlights its meaning.",
            # Creative/specific direction prompts
            lambda word: f"Craft a descriptive sentence featuring the word '{word}'.",
            lambda word: f"Compose a sentence that showcases how '{word}' can be used effectively.",
            lambda word: f"Build a sentence around the word '{word}', making it the focal point.",
            lambda word: f"Write a sentence where '{word}' plays a key role in conveying the meaning.",
            lambda word: f"Create a vivid sentence incorporating '{word}' naturally.",
            # Context-providing prompts
            lambda word: f"Imagine a scenario where '{word}' would be relevant. Write a sentence about it.",
            lambda word: f"Picture a situation involving '{word}'. Describe it in one sentence.",
            lambda word: f"Taking '{word}' as your inspiration, construct a meaningful sentence.",
            lambda word: f"Using '{word}' as the foundation, develop an interesting sentence.",
            lambda word: f"With '{word}' in mind, write a sentence that tells a mini-story.",
            # Style-specific prompts
            lambda word: f"Write a recipe for a {word}-themed dish. You must use the word '{word}'.",
            lambda word: f"Compose a short story with the theme of '{word}'.",
            lambda word: f"Please write a poem themed around the word '{word}'.",
            lambda word: f"List 5 things you know about the word '{word}'.",
            lambda word: f"Can you write lyrics to a {word}-themed song?",
        ]
        random.shuffle(self.shuffled_prompts)  # Shuffle in place

    def reset(self):
        self.last_taken_index = -1

    def get_prompt(self, word: str) -> str:
        sentence = self.shuffled_prompts[self.last_taken_index](word)
        self.last_taken_index = (self.last_taken_index + 1) % len(self.shuffled_prompts)
        # Chance of removing or replacing quotations
        if random.random() < 0.33:
            sentence = sentence.replace("'", "")
        elif random.random() < 0.66:
            sentence = sentence.replace("'", "").replace("'", "")

        # Chance of removing punctuation
        if random.random() < 0.5:
            sentence = sentence.replace(".", "").replace("!", "").replace("?", "")
        return sentence


def generate_twoword_data_for_words(
    model: HFHardPrompted,
    words: List[str],
    logger: Logger,
    num_sentences_per_word: int = 5,
    prompt_formatter: WordToPromptFormatter = TwoWordVariedDirectPromptFormatter(),
) -> Dict[Tuple[str, str], List[Tuple[str, str, List[int]]]]:
    """
    For each word in 'words', generates sentences that include that word paired with
    a randomly selected other word from the list.

    Args:
        model: The language model to use for generation
        words: List of words to generate sentences for
        logger: Logger instance for tracking progress
        num_sentences_per_word: Number of sentences to generate per word
        prompt_formatter: Formatter to create prompts for word pairs

    Returns:
        dict: { word: [(prompt, response), ...] } where each response contains both words
    """
    logger.print(
        f"Generating two-word data... ({num_sentences_per_word} sentences for each word in {words})"
    )
    data_dict = defaultdict(list)

    # Generate prompts for each word paired with random other words
    prompts = []
    word_pairs = []  # Keep track of which words each prompt is for

    num_sentences_per_pair = 1 + (
        num_sentences_per_word // (len(words) - 1 + len(words) - 1)
    )

    logger.print(
        f"Generating {num_sentences_per_pair} sentences per pair, so that we have {num_sentences_per_pair * (len(words)-1 + len(words)-1)} sentences per word total"
    )
    for _ in range(num_sentences_per_pair):
        for word1 in words:
            for word2 in words:
                if word1 == word2:
                    continue
                prompt = prompt_formatter.get_prompt(word1, word2)
                prompts.append(prompt)
                word_pairs.append((word1, word2))

    # Get responses for all prompts in one batch
    responses, token_ids = get_llm_responses(
        model, prompts, max_tokens=50, temperature=0.7, logger=logger
    )

    # Add each prompt-response pair to the first word's list
    for (word1, word2), prompt, response, token_id_tensor in zip(
        word_pairs, prompts, responses, token_ids
    ):
        response = sanitize_response(response)
        data_dict[(word1, word2)].append((prompt, response, token_id_tensor.tolist()))

    logger.optional_print(f"Generated two-word data: {data_dict}")
    return data_dict


def generate_data_for_words(
    model: HFHardPrompted,
    words: List[str],
    logger: Logger,
    num_sentences_per_word: int = 5,
    prompt_formatter: WordToPromptFormatter = PleaseWritePromptFormatter(),
) -> dict[str, List[Tuple[str, str]]]:
    """
    For each word in 'words', calls get_llm_response(...) multiple times
    with a prompt like "Please write a sentence that includes the word X."
    Returns a dict: { word: [(prompt, response), ...] }
    """
    logger.print(
        f"Generating raw data... (specifically {num_sentences_per_word} sentences for each word in {words})"
    )
    data_dict = {w: [] for w in words}

    # Generate all prompts at once
    prompts = []
    for w in words:
        prompts.extend(
            [prompt_formatter.get_prompt(w) for _ in range(num_sentences_per_word)]
        )

    # Get responses for all prompts in one batch
    responses, token_ids = get_llm_responses(
        model, prompts, max_tokens=25, temperature=0.7, logger=logger
    )

    # Distribute prompts and responses to appropriate words
    i = 0
    for w in words:
        for _ in range(num_sentences_per_word):
            prompt = prompts[i]
            response = sanitize_response(responses[i])
            data_dict[w].append((prompt, response, token_ids[i].tolist()))
            i += 1

    logger.optional_print(f"Generated data for words {words}: {data_dict}")
    return data_dict


def split_dict_values_by_key(
    dict: dict, key: str
) -> Tuple[List[Tuple[str, str, List[int]]], List[Tuple[str, str, List[int]]]]:
    pos_samples = dict[key]
    neg_samples = [s for k, v in dict.items() if k != key for s in v]
    return pos_samples, neg_samples


class WordToConditionalPrompts:
    def __init__(self):
        self.formatter_name = "Unimplemented"

    def get_dontthink_prompt(self) -> str:
        pass

    def get_prompt(self) -> str:
        pass


class DontThinkPromptFormatter(WordToConditionalPrompts):
    def __init__(self):
        self.formatter_name = "DontThinkPrompt"

    def get_dontthink_prompt(self, word: str) -> str:
        return f"You are being probed for {word}. Write a sentence about {word}."

    def get_prompt(self, word: str) -> str:
        return f"Write a sentence about {word}."


class DontThinkPleasePromptFormatter(WordToConditionalPrompts):
    def __init__(self):
        self.formatter_name = "DontThinkPleasePrompt"

    def get_dontthink_prompt(self, word: str) -> str:
        return f"You are being probed for '{word}'. Please write a sentence that includes the word '{word}'."

    def get_prompt(self, word: str) -> str:
        return f"Please write a sentence that includes the word '{word}'."

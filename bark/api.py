from typing import Optional

import numpy as np

from .generation import codec_decode, generate_coarse, generate_fine, generate_text_semantic


def text_to_semantic(
    text: str,
    history_prompt: Optional[str] = None,
    temp: float = 0,
    base = None,
    confused_travolta_mode = False,
    silent: bool = False,
):
    """Generate semantic array from text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar

    Returns:
        numpy semantic array to be fed into `semantic_to_waveform`
    """
    allow_early_stop = not confused_travolta_mode
    
    x_semantic = generate_text_semantic(
        text,
        history_prompt=history_prompt,
        temp=temp,
        base=base,
        allow_early_stop=allow_early_stop,
        silent=silent,
        use_kv_caching=True
    )
    return x_semantic


def semantic_to_waveform(
    semantic_tokens: np.ndarray,
    history_prompt: Optional[str] = None,
    temp: float = 0,
    base=None,
    use_kv_caching=True,
    silent: bool = False,
    output_full: bool = False,
):
    """Generate audio array from semantic input.

    Args:
        semantic_tokens: semantic token output from `text_to_semantic`
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    x_coarse_gen = generate_coarse(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=temp,
        base=base,
        silent=silent,
        use_kv_caching=use_kv_caching
    )
    x_fine_gen = generate_fine(
        x_coarse_gen,
        history_prompt=history_prompt,
        temp=0.5,
        base=base,
        silent=silent
    )
    audio_arr = codec_decode(x_fine_gen)
    return audio_arr, x_coarse_gen, x_fine_gen

def save_as_prompt(filepath, full_generation):
    assert(filepath.endswith(".npz"))
    assert(isinstance(full_generation, dict))
    assert("semantic_prompt" in full_generation)
    assert("coarse_prompt" in full_generation)
    assert("fine_prompt" in full_generation)
    np.savez(filepath, **full_generation)


def generate_audio(
    text: str,
    history_prompt: Optional[str] = None,
    text_temp: float = 0,
    base = None,
    confused_travolta_mode = False,
    silent: bool = False,
    output_full: bool = False,
):
    """Generate audio array from input text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    x_semantic = text_to_semantic(
        text,
        history_prompt=history_prompt,
        temp=text_temp,
        silent=silent,
        base=base,
        confused_travolta_mode=confused_travolta_mode
        )
    audio_arr, c, f = semantic_to_waveform(x_semantic, history_prompt=history_prompt, temp=text_temp, base=base)
    return audio_arr, [x_semantic, c, f]
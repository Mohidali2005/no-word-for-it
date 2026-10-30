import pytest
from transformers import AutoProcessor

from nwf import config
from nwf import run_items

# token ids checked by hand on the pinned tokenizers before any model data
EXPECTED = {"qwen":([32,33,34,35],320),"gemma":([236776,236799,236780,236796],568)}


@pytest.mark.parametrize("name",["qwen","gemma"])
def test_letters_are_single_tokens_after_the_answer_start(name):
    spec = config.MODELS[name]
    processor = AutoProcessor.from_pretrained(spec["repo"],revision=spec["revision"],use_fast=config.FAMILIES[name]["use_fast"])
    letters,last = EXPECTED[name]
    assert run_items.letter_ids(processor.tokenizer)==letters
    text = processor.apply_chat_template([{"role":"user","content":[{"type":"text","text":config.PROMPT}]}],tokenize=False,add_generation_prompt=True)+config.PREFILL
    assert processor.tokenizer(text,add_special_tokens=False)["input_ids"][-1]==last

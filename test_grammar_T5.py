#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:34:01 2024

@author: cringwal
"""

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer
from rdflib import Graph
import json
import torch
import sys

from transformers_cfg.parser import parse_ebnf

from transformers_cfg.grammar_utils import IncrementalGrammarConstraint 
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers_cfg.recognizer import StringRecognizer

# Detect if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
def clean_and_load_CKPT(ckpt):
    
    checkpoint=torch.load(ckpt,map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in checkpoint["state_dict"].items():
        name = k.replace("model.", "") # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

############# PATH OF FILES AND 
model_name="t5-base"
tkn_path="/user/cringwal/home/Desktop/Models_SAVE/12ShadesOfSyntax/T5_b_v_t_1/experiments/experiments/T5-small_tokenizer"
cpt="/user/cringwal/home/Desktop/Models_SAVE/12ShadesOfSyntax/T5_b_v_t_1/experiments/240ShadesOfSyntaxT5_bf16/DS_turtleS_1inLine_1facto_T5_base_t5-base_2-epoch=08-val_loss=0.02.ckpt"


checkpoint=torch.load(cpt,map_location=torch.device('cpu'))


##########################  LOAD MODEL AND TOKENIZER

tokenizer_kwargs = {
    "use_fast": True,
#    "add_tokens": all_vocab
}

tokenizer = T5Tokenizer.from_pretrained(
            tkn_path,
            #**tokenizer_kwargs
        )

config = AutoConfig.from_pretrained(
        model_name,
        decoder_start_token_id = 0,
        early_stopping = False,
        no_repeat_ngram_size = 0,
        dropout=0.1,
        forced_bos_token_id=None,
    )


model = AutoModelForSeq2SeqLM.from_config(
        config=config
    )

model.resize_token_embeddings(len(tokenizer))

ckpt=clean_and_load_CKPT(cpt)

model.load_state_dict(ckpt)


############################

example = ["<s>Erwan Falwad (31 July 1730 – 28 June 1795), also known as The Falder, was an American philosopher and politician. During his career that spanned twenty years, he wrote a variety of theories about the supremacy of people with a first name starting with the letter F. (...).</s>",
           "<s>Translate English to Turtle : [Erwan_Falwad] Erwan Falwad (31 July 1730 – 28 June 1795), also known as The Falder, was an American philosopher and politician. During his career that spanned twenty years, he wrote a variety of theories about the supremacy of people with a first name starting with the letter F. (...).</s>",
          "<s>Translate English to Turtle : Erwan Falwad (31 July 1730 – 28 June 1795), also known as The Falder, was an American philosopher and politician. During his career that spanned twenty years, he wrote a variety of theories about the supremacy of people with a first name starting with the letter F. (...).</s>",
          "<s>Translate English to Turtle : [<extra_id_0>] Erwan Falwad (31 July 1730 – 28 June 1795), also known as The Falder, was an American philosopher and politician. During his career that spanned twenty years, he wrote a variety of theories about the supremacy of people with a first name starting with the letter F. (...).</s>",
          "<s>Translate English to Turtle : [] Erwan Falwad (31 July 1730 – 28 June 1795), also known as The Falder, was an American philosopher and politician. During his career that spanned twenty years, he wrote a variety of theories about the supremacy of people with a first name starting with the letter F. (...).</s>",
          "<s>Translate English to Turtle : [Evelyn_Sterling] Dr. Evelyn Sterling was born on July 12, 1978, in the quaint town of Oakridge, nestled in the hills of Vermont, USA. From a young age, she exhibited a keen interest in the stars, often spending nights gazing at the celestial wonders with her father's telescope. Her fascination with the cosmos grew as she devoured books on astronomy and physics.</s>"]
example=["<s>Translate English to Turtle : [Evelyn_Sterling] Dr. Evelyn Sterling was born on July 12, 1978, in the quaint town of Oakridge, nestled in the hills of Vermont, USA. From a young age, she exhibited a keen interest in the stars, often spending nights gazing at the celestial wonders with her father's telescope. Her fascination with the cosmos grew as she devoured books on astronomy and physics.</s>"]
inputs = tokenizer(example, return_tensors='pt', padding=True, truncation=True)
gen_kwargs = {
            "max_length": 512,
            "early_stopping": False,
            "length_penalty": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": 3
        }

translated_tokens = model.generate(inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            #use_cache = False,
            **gen_kwargs)
generations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True, spaces_between_special_tokens = True)
print(generations)
############################ OK WITH TOKENIZER

from transformers_cfg.parser import parse_ebnf

from transformers_cfg.grammar_utils import IncrementalGrammarConstraint 
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers_cfg.recognizer import StringRecognizer


with open("/user/cringwal/home/Desktop/turtle_light_facto2.ebnf", "r") as file:
    grammar_str = file.read()

eos_token = " ."
eos_token_id = tokenizer.encode(eos_token, padding=True, spaces_between_special_tokens = True)["input_ids"]


grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

# Generate
input_ids = tokenizer(example, return_tensors="pt", padding=True)["input_ids"]

gen_kwargs = {
            "max_length": 512,
            "early_stopping": False,
            "length_penalty": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": 3
        }



output = model.generate(
    input_ids,
    logits_processor=[grammar_processor],
    #repetition_penalty=1.1,
    #num_return_sequences=1,
    **gen_kwargs
)
# decode output
generations = tokenizer.batch_decode(output, skip_special_tokens=True, spaces_between_special_tokens = True)
print(generations)

################## NOW TEST IT 
with open("/user/cringwal/home/Desktop/turtle_light_facto2.ebnf", "r") as file:
    grammar_str = file.read()
parsed_grammar = parse_ebnf(grammar_str)

start_rule_id = parsed_grammar.symbol_table["root"]
recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

for gen in generations:
    print("------------------GENERATED")
    print(gen)
    print("------------------------------> accepted ? ")
    is_accepted = recognizer._accept_prefix(gen)
    print(">>>>> ",is_accepted)

gen2=':Evelyn_Sterling a :Person; :label "Evelyn Sterling"; :birthDate "1978-07-12"; :birthYear "1978".'
is_accepted = recognizer._accept_prefix(gen2)
print(">>>>> ",is_accepted)
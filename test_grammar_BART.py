#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:34:01 2024

@author: cringwal
"""
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
import json
import torch

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
        name = k.replace("model.model", "model") # remove `module.`
        if(k in ["model.final_logits_bias","model.lm_head.weight"]):
            name = k.replace("model.", "")
        new_state_dict[name] = v
    return new_state_dict


############# PATH OF FILES AND 
model_name="facebook/bart-base"
tkn_path="/user/cringwal/home/Desktop/Models_SAVE/12ShadesOfSyntax/B_b_v_t_1_f/experiments/experiments/BART-base_tokenizer"
cpt="/user/cringwal/home/Desktop/Models_SAVE/12ShadesOfSyntax/B_b_v_t_1_f/experiments/240ShadesOfSyntax/last.ckpt"



##########################  LOAD MODEL AND TOKENIZER

tokenizer_kwargs = {
    "use_fast": True,
#    "add_tokens": all_vocab
}
tokenizer = AutoTokenizer.from_pretrained(
            tkn_path,
            **tokenizer_kwargs
        )



config = AutoConfig.from_pretrained(
        model_name,
        decoder_start_token_id = 0,
        #early_stopping = False,
        no_repeat_ngram_size = 0,
        dropout=0.1,
        forced_bos_token_id=None,
    )

model = AutoModelForSeq2SeqLM.from_config(
        config=config
    )

model.resize_token_embeddings(len(tokenizer))

ckpt=clean_and_load_CKPT(cpt)

model.load_state_dict( ckpt)


############################ TEST IT WITH 

example = ["<s>Percy_Hynes_White : Percy Hynes White (born October 8, 2001) is a Canadian actor. He is known for his roles in films such as Edge of Winter and A Christmas Horror Story, for his role in the television series Between, and for his starring role as Andy Strucker in The Gifted.</s>",
           "<s>Evelyn_Sterling : Dr. Evelyn Sterling was born on July 12, 1978, in the quaint town of Oakridge, nestled in the hills of Vermont, USA. From a young age, she exhibited a keen interest in the stars, often spending nights gazing at the celestial wonders with her father's telescope. Her fascination with the cosmos grew as she devoured books on astronomy and physics.</s>"]

inputs = tokenizer(example, return_tensors='pt', padding=True, truncation=True)
gen_kwargs = {
            "max_length": 1024,
            "early_stopping": False,
            "length_penalty": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": 2
        }

translated_tokens = model.generate(inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            use_cache = True,
            **gen_kwargs)
print( tokenizer.batch_decode(translated_tokens, skip_special_tokens=True, spaces_between_special_tokens = True))

############################ OK WITH TOKENIZER

from transformers_cfg.parser import parse_ebnf

from transformers_cfg.grammar_utils import IncrementalGrammarConstraint 
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers_cfg.recognizer import StringRecognizer


with open("/user/cringwal/home/Desktop/turtle_light_facto2.ebnf", "r") as file:
    grammar_str = file.read()


grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
grammar_processor = GrammarConstrainedLogitsProcessor(grammar)


# Generate
input_ids = tokenizer(example, return_tensors="pt", padding=True)["input_ids"]


gen_kwargs = {
           "max_length": 1024,
           "early_stopping": False,
           "length_penalty": 0,
           "no_repeat_ngram_size": 0,
           "num_beams": 2
        }



output = model.generate(
    input_ids,
    logits_processor=[grammar_processor],
    #repetition_penalty=1.1,
    #num_return_sequences=1,
    **gen_kwargs
)
# decode output
generations = tokenizer.batch_decode(output, skip_special_tokens=True)
print(generations)

################## NOW TEST IT 

parsed_grammar = parse_ebnf(grammar_str)

start_rule_id = parsed_grammar.symbol_table["root"]
recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

for gen in generations:
    print("------------------GENERATED")
    print(gen)
    print("------------------------------> accepted ? ")
    is_accepted = recognizer._accept_prefix(gen)
    print(">>>>> ",is_accepted)
    
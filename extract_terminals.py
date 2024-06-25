#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:12:23 2024

@author: cringwal
"""


from transformers_cfg.parser import parse_ebnf

with open("/user/cringwal/home/Desktop/turtle_light_facto2.ebnf", "r") as file:
    grammar_str = file.read()
parsed_grammar = parse_ebnf(grammar_str)

e=grammar_str.split("\n")
import re
terminals=set()
for st in e:
    list_f=re.findall(r'"(.*?(?<!\\))"', st)
    for l in list_f:
        terminals.add(l)
    
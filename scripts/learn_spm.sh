#!/bin/bash

module use -a /projappl/nlpl/software/modules/etc
module load nlpl-mttools

INPUT=$1 
MOD_PREFIX=$2

spm_train --input=$INPUT --model_prefix=$MOD_PREFIX --vocab_size=32000 --character_coverage=1.0 

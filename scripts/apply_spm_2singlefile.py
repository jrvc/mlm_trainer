import sentencepiece as spm
from pathlib import Path
import os
import sys

# USAGE:
#       srun -p test --time=00:15:00 --mem-per-cpu=2G --account=project_2001970  \
#             python /projappl/project_2001970/amerNLP2021_HY/scripts/apply_spm_2singlefile.py infile SPmodel outfile


infile = sys.argv[1] #"/scratch/project_2001970/americasnlp2021/data/processed_data"
modelSP = sys.argv[2] #f'{folder}/../SP_amernlp.model'
output = sys.argv[3] 

#modelSP ="/scratch/project_2001970/americasnlp2021/data//SP_amernlp.model"

sp = spm.SentencePieceProcessor(model_file=modelSP)


print(f'processing: {infile}')                          
fileOUT = open(output, "wt")

with open(infile,'rt') as f:
    for line in f:
        encodedText = sp.encode(line, out_type=str)
        stringT=' '.join(encodedText)
        fileOUT.write(stringT+'\n')

print(f'ouput: {output}')
fileOUT.close()


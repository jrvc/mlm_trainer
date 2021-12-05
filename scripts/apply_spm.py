import sentencepiece as spm
from pathlib import Path
import os
import sys

# USAGE:
#       srun -p test --time=00:15:00 --mem-per-cpu=2G --account=project_2001970  python /projappl/project_2001970/amerNLP2021_HY/scripts/apply_spm.py


folder = sys.argv[1] #"/scratch/project_2001970/americasnlp2021/data/processed_data"
modelSP = sys.argv[2] #f'{folder}/../SP_amernlp.model'

#modelSP ="/scratch/project_2001970/americasnlp2021/data//SP_amernlp.model"

sp = spm.SentencePieceProcessor(model_file=modelSP)
if not 'multiling' in Path(modelSP).name:
    for path in Path(folder).rglob('*'):
        isDirectory = os.path.isdir(path)
        if not isDirectory and not path.suffix=='.ids':
            print(f'processing: {path}')                          
            pathOUT = path.with_name(f'{path.name}.sp')
            fileOUT = open(pathOUT, "wt")

            with open(path,'rt') as f:
                for line in f:
                    encodedText = sp.encode(line, out_type=str)
                    stringT=' '.join(encodedText)
                    fileOUT.write(stringT+'\n')

            print(f'ouput: {pathOUT}')
                    
            fileOUT.close()
else:
    for path in Path(folder).rglob('en-??/*'):
        if path.suffix in ['.en', '.de', '.fi', '.et']:
            print(f'processing: {path}')                          
            pathOUT = path.parent.with_name('multiling').joinpath(f'{path.name}.sp')
            fileOUT = open(pathOUT, "wt")

            with open(path,'rt') as f:
                for line in f:
                    encodedText = sp.encode(line, out_type=str)
                    stringT=' '.join(encodedText)
                    fileOUT.write(stringT+'\n')

            print(f'ouput: {pathOUT}')
                    
            fileOUT.close()


# apply spm to test sets
if False:
    sp = spm.SentencePieceProcessor(model_file=modelSP)
    for path in Path(folder).rglob('*'):
        if path.name=='test.es':
            pathOUT = path.with_name(f'{path.stem}.es.sp')
            print(pathOUT)

            fileOUT = open(pathOUT, "wt")
            with open(path,'rt') as f:
                for line in f:
                    encodedText = sp.encode(line, out_type=str)
                    stringT=' '.join(encodedText)
                    fileOUT.write(stringT+'\n')
                    
            fileOUT.close()
# apply spm to analysis test set
if False:       
    path=Path('/projappl/project_2001970/amerNLP2021_HY/testset/AmericasNLP_Analysis-1.dms')
    sp = spm.SentencePieceProcessor(model_file=modelSP)
        
    pathOUT = path.with_name(f'{path.stem}.es.sp')
    print(pathOUT)

    fileOUT = open(pathOUT, "wt")
    with open(path,'rt') as f:
        for line in f:
            encodedText = sp.encode(line, out_type=str)
            stringT=' '.join(encodedText)
            fileOUT.write(stringT+'\n')
                
    fileOUT.close()

# apply spm to analysis test set 2
if False:       
    path=Path('/projappl/project_2001970/amerNLP2021_HY/testset/AmericasNLP_Analysis-2')
    sp = spm.SentencePieceProcessor(model_file=modelSP)
        
    pathOUT = path.with_name(f'{path.stem}.es.sp')
    print(pathOUT)

    fileOUT = open(pathOUT, "wt")
    with open(path,'rt') as f:
        for line in f:
            encodedText = sp.encode(line, out_type=str)
            stringT=' '.join(encodedText)
            fileOUT.write(stringT+'\n')
                
    fileOUT.close()



    
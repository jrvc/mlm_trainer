#!/usr/bin/env python
# coding: utf-8
'''
	this code has been adapted form James Briggs' code
'''
# ## Training Tokenizer <- SKIP: data has been tokenized.


import tokenizers 
import torch
import random
import json
import argparse

from pathlib import Path
from tokenizers.processors import BertProcessing
from tqdm.auto import tqdm
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import BertConfig
from transformers import BertLMHeadModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

#from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter


'''
python tok-and-trainer_simple_bert-from-scratch.py \
 --batch_size 16 --max_length=50 --train_epochs 4 --lca_steps 25 \
 --n_hiddenlayers 12 --vocab_size 32000 --warmup_steps 8000 \
 --train_data_file data/corpora/en-de/ \
 --output_dir ./simpleBERTfromScratch/ \
 --overwrite_cache --use_cuda 

SRCLANG='en'

TOKPATH=f'{DATAPATH}/tok_{SRCLANG}'
SAVEMODEL="./simpleBERTfromScratch/"
overwrite_cache = False
USECUDA=True
'''

def train_tokenizer(data_path:str, tokenizer_path:str, vocabsz:int=32000):
    # initialize and train the tokenizer. Use BERT special tokens.
    if Path(data_path).is_dir():
        paths = [str(x) for x in Path(data_path).glob(f'*train*')]
    else:
        paths = [data_path]
    tokenizer = tokenizers.ByteLevelBPETokenizer()
    # and train
    tokenizer.train(files=paths, vocab_size=vocabsz, min_frequency=2,
                    show_progress=True,
                    special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

    Path(tokenizer_path).mkdir(exist_ok=True, parents=True)
    print(f'Saving tokenizer: {tokenizer_path}')
    return tokenizer.save_model(tokenizer_path)

def load_tokenizer(tokenizer_path:str):
    # ## Load Tokenizer
    from transformers import RobertaTokenizerFast
    return RobertaTokenizerFast.from_pretrained(tokenizer_path)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, 
        data:[str,list], 
        vocab:str, 
        merges:str, 
        max_length:int=512, 
        srclang:str='en',
        mlm_prob:float=0.15):
        # initialize the tokenizer using the tokenizer we initialized and saved to file
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(vocab, merges)
        self.tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", self.tokenizer.token_to_id("</s>")),
            ("<s>", self.tokenizer.token_to_id("<s>")),
        )
        # truncate anything more than 512 tokens in length
        self.tokenizer.enable_truncation(max_length=max_length)
        # and enable padding to max_length too
        self.tokenizer.enable_padding(length=max_length, pad_token="<pad>", pad_id=1)
        self.paths = self.get_paths(data)
        
        # open the first file to get 'expected' length
        with open(self.paths[0], 'r', encoding='utf-8') as fp:
            lines = fp.read().split('\n')
        # save file length as 'expected' length
        self.file_size = len(lines)
        self.mlm_prob = mlm_prob
    
    def __len__(self):
        # we calculate the total number of examples as the number of samples in the
        # first file, multipled by the number of files, minus the final value
        length = self.file_size * len(self.paths) - self.file_size
        with open(self.paths[-1], 'r', encoding='utf-8') as fp:
            lines = fp.read().split("\n")
        length += len(lines)
        return length
    
    def __getitem__(self, i):
        # get the file number and sample number based on i
        file_i, sample_i = self.get_loc(i)
        # load file
        with open(self.paths[file_i], 'r', encoding='utf-8') as fp:
            lines = fp.read().split("\n")
        # extract required sample
        sample = lines[sample_i]
        # encode
        sample = self.tokenizer.encode(sample)
        # convert tokens to tensor
        try:
            targets = torch.tensor(sample.ids)
        except RuntimeError:
            raise RuntimeError(f"{sample=}")
        # create attention mask tensor
        mask = torch.tensor(sample.attention_mask)
        # mask ~15% of tokens to create inputs
        input_ids = self.mlm(targets.detach().clone())
        # return dictionary of input_ids, attention_mask, and labels
        return {'input_ids': input_ids, 'attention_mask': mask, 'labels': targets}

    def get_loc(self, i):
        # get file number
        file_num = int(i / self.file_size)
        sample_num = i % self.file_size
        return file_num, sample_num
    
    def get_paths(self, datapath:[str,list]):
        if isinstance(datapath,list):
            paths = [x for x in datapath]
        else:
            if Path(datapath).is_dir():
                #TODO: enable giving a directory and choosign all the shards of the train data
                paths = [str(x) for x in Path(datapath).glob(f'**/*train*')] # TODO: does this work? 
                # reorder paths (above will give *.xaa as first and *.xzz as final file)
                paths.sort()
            elif Path(datapath).is_file():
                paths = [datapath]
                
        if len(paths) == 0: 
            RuntimeError('given data path is broken or this is wrong :P check routine')
        return paths
         
    def mlm(self, tensor):
        # create random array of floats with equal dims to tensor
        rand = torch.rand(tensor.shape)
        # mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
        mask_arr = (rand < self.mlm_prob) * (tensor != 0) * (tensor != 1) * (tensor != 2)
        # get indices of mask positions from mask array
        mask_idx = torch.flatten(mask_arr.nonzero()).tolist()
        # mask tensor and return
        tensor[mask_idx] = 4
        return tensor



def main(args):
    EXPNAME = Path(args.train_data_file).name
    TOKPATH = f'{args.output_dir}/{EXPNAME}_tokenizer'
    
    EXPNAME = f'{EXPNAME}_{args.n_hiddenlayers}layers'
    TBwriter =  f'{args.output_dir}/{EXPNAME}_tbwriter'

    print(args) # keep experims record 

    Path(f'{args.output_dir}').mkdir(exist_ok=True, parents=True)

    if Path(f'{TOKPATH}/vocab.json').is_file() and not args.overwrite_cache:
        print(f'Catched tokenizer form {TOKPATH}')
        vocab= f'{TOKPATH}/vocab.json'
        subwrdmod= f'{TOKPATH}/merges.txt'
    else:
        print('Training tokenizer')
        vocab, subwrdmod = train_tokenizer(args.train_data_file, TOKPATH, args.vocab_size)
    
    print('Initializing dataset and dataloader')
    dataset = Dataset(
        args.train_data_file, 
        vocab, 
        subwrdmod, 
        args.max_length
        )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Import and initialize a BERT model with a language modeling head.
    print('Initializing model')
    config = BertConfig(
        vocab_size=dataset.tokenizer.get_vocab_size(),  
        hidden_size=768,
        num_hidden_layers=args.n_hiddenlayers,
        pad_token_id=0
    )
    model = BertLMHeadModel(config)
    if args.use_roberta:
        config = RobertaConfig(
            vocab_size=dataset.tokenizer.get_vocab_size(),
            max_position_embeddings=2+args.max_length,
            num_hidden_layers=args.n_hiddenlayers,
            type_vocab_type=1,
            )
        model = RobertaForMaskedLM(config)

    modelname=str(type(model)).strip("\'>").split(".")[-1]
    # move onto training
    device = torch.device('cuda') if (torch.cuda.is_available() and args.use_cuda)else torch.device('cpu')
    model.to(device)
    model.train()

    # Init Adam with weighted decay optimizer
    t_total = args.train_epochs * len(loader) + 500
    optim = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # TRAINING LOOP:
    print(f'Initializing tensorboard writer: {TBwriter}')
    writer = SummaryWriter(TBwriter)

    print(f'Initizating training routine. ')
    print(f'      Model to use: \t \t {modelname} with {args.n_hiddenlayers} layers')
    print(f'      Datasets to use: \t \t {len(dataset.paths)}')
    if len(dataset.paths) > 5:
        print(f'        first 3 of `em: \t {dataset.paths[:3]}')
    else:
        print(f'                        \t {dataset.paths}')
    print(f'      Number of examples: \t {len(dataset)}')
    print(f'      Epochs: \t \t \t {args.train_epochs}')
    print(f'      Batch size:  \t \t {args.batch_size}')
    print(f'      Steps/epoch: \t \t {len(loader)}')
    print(f'      Truncating after: \t {dataset.tokenizer.truncation["max_length"]} tokens')
    print(f'      Using GPU: \t \t {args.use_cuda}')
    if args.lca_steps > 0:
        print(f'      Logging LCA every: \t {args.lca_steps} steps')
    print(f'      Warmup steps: \t {args.warmup_steps}')
    print(f'      Optimizer: {optim}', flush=True)
    # LCA
    lca_logs = {k: dict() for k, v in model.named_parameters() if v.requires_grad}
    lca_params = {k: torch.zeros_like(v.data) for k, v in model.named_parameters() if v.requires_grad}
    lca_interv = {k: torch.zeros_like(v.data) for k, v in model.named_parameters() if v.requires_grad}
    # \LCA
    step = 0
    for epoch in range(args.train_epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=True)
        for batch in loop:
            #LCA
            theta_t = {k: v.data.clone() for k, v in model.named_parameters() if v.requires_grad}
            # \LCA

            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)
            # extract loss
            loss = outputs.loss
            # take loss for tensorboard
            writer.add_scalar('Loss/train', loss, step)
            writer.add_scalar("learning rate", scheduler.get_last_lr()[0], step)
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            scheduler.step()
            # LCA
            if args.lca_steps > 0:
                for k, v in model.named_parameters():
                    if not v.requires_grad or isinstance(v.grad, type(None)):
                        continue
                    lca_params[k] = (v.data - theta_t[k]) * v.grad
                    lca_interv[k] += lca_params[k]

                    if step % args.lca_steps == 0:
                        for k, v in lca_params.items():
                            lca_sum = v.sum().item()
                            lca_mean = v.mean().item()
                            lca_interval_mean = (lca_interv[k] / args.lca_steps).mean().item()
                            lca_interval_sum = (lca_interv[k] / args.lca_steps).sum().item()
                            lca_logs[k][f'STEP_{step}'] = {'sum': lca_sum, 'mean': lca_mean,
                                                           'interval.sum': lca_interval_sum,
                                                           'interval.mean': lca_interval_mean}

                    # log these 10 times per epoch
                    if (step % int(len(loader)/10) ) == 0:
                        opath = Path(f'{args.output_dir}/{EXPNAME}_lca_logs.json')
                        with open(opath, 'w+') as f:
                            json.dump(lca_logs, f)
            #\LCA
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
            step += 1

        # https://huggingface.co/blog/how-to-train

        print(f'Saving model ckpt: {args.output_dir}')
        model.save_pretrained(f'{args.output_dir}/{EXPNAME}_epoch{epoch}.ckpt')

    len(loop)

    
def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", 
            default=None, type=str, required=True,  
            help="The input training data file (a text file or a directory).")
    parser.add_argument("--output_dir", 
            type=str, required=True,
            help="Where all outputs will be saved",)
    # Other parameters
    parser.add_argument("--srclang", 
            default=None, type=str,)
    parser.add_argument("--eval_data_file", 
            default=None, type=str,
            help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--should_continue", 
            action="store_true", 
            help="Whether to continue from latest checkpoint in output_dir")
    parser.add_argument("--mlm_probability", 
            type=float, default=0.15, 
            help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--max_length", 
            default=512, type=int)
    parser.add_argument("--vocab_size", 
            default=32000, type=int,
            help="BPE vocab size, i.e., number of splits.")
    parser.add_argument("--batch_size", 
            default=512, type=int,
            help="Batch size per for training.")
    parser.add_argument("--n_hiddenlayers", 
            default=12, type=int,
            help="Number of hidden layers of BERT.")
    parser.add_argument("--eval_batch_size" , 
            default=256, type=int, 
            help="Batch size per for evaluation.")
    parser.add_argument("--train_epochs", 
            default=100, type=int, 
            help="Total number of training epochs to perform.")
    parser.add_argument("--lca_steps", 
            type=int, default=-1)
    parser.add_argument("--save_steps", 
            type=int, default=500, 
            help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", 
            action="store_true")
    parser.add_argument("--use_cuda", 
            action="store_true")
    parser.add_argument("--overwrite_output_dir", 
            action="store_true")
    parser.add_argument("--use_roberta", 
            action="store_true",
            help="use a roberta model instead of BERT")
    parser.add_argument("--overwrite_cache", 
            action="store_true")
    parser.add_argument("--seed", 
            type=int, default=42, 
            help="random seed for initialization")
    parser.add_argument("--warmup_steps",
            default=50, type=int, 
            help="Linear warmup over warmup_steps.")
    parser.add_argument("--learning_rate", 
        default=2e-3, type=float, 
        help="The initial learning rate for AdamW.")
    #parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    #parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    #parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    #parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    #parser.add_argument("--max_steps", default=-1, type=int)


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseargs()
    main(args)
    

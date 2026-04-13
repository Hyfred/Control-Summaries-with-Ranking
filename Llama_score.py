
import torch
import os
import torch
import torch.nn as nn
# import traceback
# from unsloth.chat_templates import get_chat_template
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["UNSLOTH_RETURN_LOGITS"] = "0,1"
tqdm.pandas()

class LLaMAScorer:
    def __init__(self, device='cuda:1', max_length=1024, checkpoint=''):
        # Set up model
        self.device = device
        print(self.device)
        self.max_length = max_length
        print('max_length: ', max_length)
        if 'Llama' in checkpoint:
            print('Llama model')
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                ).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                ).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.log_softmax = nn.LogSoftmax(dim=2)  
        self.loss_fn = nn.NLLLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
        self.model.eval()

    def NLL_loss(self, logits, labels):
        batch_size, seq_len, num_classes = logits.size()
        logits = logits.view(-1, num_classes) # [batch_size * seq_len, num_classes]

        # Get a mask for tokens that are not `ignore_index`
        valid_mask = (labels != self.tokenizer.pad_token_id).float()  # Shape: [batch_size, length]
        # Count the number of valid tokens per sentence
        valid_token_counts = valid_mask.sum(dim=1)  # Shape: [batch_size]

        labels = labels.view(-1) # [batch_size * seq_len]
        loss = self.loss_fn(logits, labels)
        loss_per_sequence = loss.view(batch_size, seq_len).sum(dim=-1)

        # Avoid division by zero (if there are sentences with no valid tokens)
        valid_token_counts = torch.clamp(valid_token_counts, min=1)
        # Compute the mean NLL for each sentence
        NLL_avg = loss_per_sequence/ valid_token_counts  # Shape: [batch_size]

        return NLL_avg

    def score(self, srcs, tgts, batch_size=14):
        """ Score a batch of examples """

        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]

            srcs_inputs = self.tokenizer.apply_chat_template(src_list, tokenize = True, return_dict=True, add_generation_prompt = True, max_length=self.max_length, truncation=True, padding="max_length", return_tensors = "pt").to(self.model.device) # for unsloth
            tgts_outputs = self.tokenizer.apply_chat_template(tgt_list, tokenize = True, return_dict=True, max_length=self.max_length, truncation=True, padding="max_length", return_tensors = "pt").to(self.model.device) # for unsloth
            # print('breakpoint:check prefix of src and tgt are the same; pass')

            try:
                with torch.no_grad():
                    outputs = self.model(
                        **srcs_inputs,
                        # labels=tgts_outputs['input_ids'],
                    )
                logits = outputs['logits']
                log_probs = self.log_softmax(logits)
                
                loss = self.NLL_loss(log_probs, tgts_outputs['input_ids'])
                score = -loss
                score_list.extend(score.tolist())
                # print('score: ',score_list)
            except RuntimeError:
                # traceback.print_exc()
                print('input_ids: ',srcs_inputs)
                print('output_ids: ', tgts_outputs)
                print(f'source: {srcs}')
                print(f'target: {tgts}')
                # exit(0)
        return score_list

    def input_chat_template(self, document, summary=None):
        # instruction = "Please summarize the input document."
        # instruction = '''Please summarize the input document, prioritizing completeness over conciseness.'''
        # instruction = '''Please summarize the input document, prioritizing conciseness over completeness.'''
        instruction = '''Please summarize the input document, balancing completeness with conciseness.'''
        if summary == None:
            row_json = [{"role": "user", "content": f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{document}\n\n### Response:\n\n"}]
        else:
            row_json = [{"role": "user", "content": f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{document}\n\n### Response:\n\n"},{"role": "assistant", "content":summary}]
        return row_json
    
    def test(self, batch_size=3):
        """ Test """
        document = ''''''
        summary = ''''''

        src_list = [
            document,
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            summary,
            "What's the problem?",
            'He is trustworthy.'
        ]

        src_input = [self.input_chat_template(doc) for doc in src_list]
        tgt_input = [self.input_chat_template(doc,summ) for doc,summ in zip(src_list,tgt_list)]

        print(self.score(src_input, tgt_input, batch_size))


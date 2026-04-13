
import re
from utils import get_keyfact_prompt, get_keyfact_alighment_prompt, parsing_llm_keyfact_output, parsing_llm_keyfact_alighment_output
from utils import compute_completeness_percentage_score, compute_conciseness_percentage_score
import os
import transformers
from utils import get_keyfact_alighment_prompt, parsing_llm_keyfact_alighment_output
from utils import compute_completeness_percentage_score, compute_conciseness_percentage_score
import pandas as pd
import torch
from openai import OpenAI
import ast
from tqdm import tqdm
import numpy as np

tqdm.pandas()

class LLamaScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint=''):
        # Load model and tokenizer
        if device == 'cuda:0':
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        else: os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        
        self.checkpoint = checkpoint

        if 'unsloth' in self.checkpoint:
            model_id = ""
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
        elif 'chatgpt' in self.checkpoint:
            _api_key = ''
            self.client = OpenAI(api_key=_api_key)

    def format_chat_template(self, prompt):
        # instruction = "Please summarize the document."  # Instruction template
        # row_json = [
        #     {"role": "system", "content": instruction},
        #     {"role": "user", "content": f"Document: {document}"}
        # ]
        row_json = [
            {"role": "user", "content": prompt}
        ]
        return row_json

    def keyfact_extraction(self, document):
        # Prepare inputs for batch processing
        data_prompt = [self.format_chat_template(get_keyfact_prompt(item)) for item in document]

        if 'unsloth' in self.checkpoint:
            outputs = self.pipeline(
                data_prompt,
                max_new_tokens=2048,
                )
            output_sent = [outputs[i][0]["generated_text"][-1]['content'] for i in range(len(outputs))]
        elif 'chatgpt' in self.checkpoint:
            output_sent = []
            for single_prompt in data_prompt:
                completion = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages = single_prompt
                )
                output_sent.append(completion.choices[0].message.content)
        keyfact = [parsing_llm_keyfact_output(out) for out in output_sent]
        return keyfact

    def keyfact_aligment(self, summary, keyfact_extra):
        # batch_sentences_old = [sent.split('.') for sent in summary]
        batch_sentences = [re.split(r'(?<=\.)\s+', para) for para in summary]
        length_sents = [len(sent) for sent in batch_sentences]
        # Prepare inputs for batch processing
        data_prompt = [self.format_chat_template(get_keyfact_alighment_prompt(keyfact,sum_sent)) for keyfact, sum_sent in zip(keyfact_extra, batch_sentences)]

        # if 'unsloth' in self.checkpoint:
        #     outputs = self.pipeline(
        #         data_prompt,
        #         max_new_tokens=2048,
        #         )
        #     output_sent = [outputs[i][0]["generated_text"][-1]['content'] for i in range(len(outputs))]

        output_sent = []
        batch_pred_alignment_labels, batch_pred_sentence_line_numbers = [], []

        for single_prompt in data_prompt:
            retry_count = 0

            while retry_count < 3:
                completion = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=single_prompt
                )
                output_content = completion.choices[0].message.content
                output_sent.append(output_content)

                # Parse the output
                pred_alignment_labels, pred_sentence_line_numbers = parsing_llm_keyfact_alighment_output(output_content)

                # If parsing results are valid, break out of retry loop
                if pred_alignment_labels or pred_sentence_line_numbers:
                    batch_pred_alignment_labels.append(pred_alignment_labels)
                    batch_pred_sentence_line_numbers.append(pred_sentence_line_numbers)
                    break  # Exit retry loop if parsing is successful

                retry_count += 1  # Increment retry counter

            # If max retries are exhausted and still empty, append empty lists
            if retry_count == 3:
                batch_pred_alignment_labels.append([])
                batch_pred_sentence_line_numbers.append([])
        
        return batch_pred_alignment_labels, batch_pred_sentence_line_numbers, length_sents

    def compute_score(self, batch_pred_alignment_labels, batch_pred_sentence_line_numbers, batch_sentences):
        completeness_score = [compute_completeness_percentage_score(pred_alignment_label) for pred_alignment_label in batch_pred_alignment_labels]
        conciseness_score = [compute_conciseness_percentage_score(pred_sentence_line_number, sent_len) for pred_sentence_line_number, sent_len in zip(batch_pred_sentence_line_numbers, batch_sentences)]
        return completeness_score, conciseness_score

    def keyfact_to_score(self, doc_list, summary_list, keyfact=None):
        if keyfact:
            keyfact_extra = keyfact
        else:
            keyfact_extra = self.keyfact_extraction(doc_list)
        if keyfact_extra:
            batch_pred_alignment_labels, batch_pred_sentence_line_numbers, batch_sentences = self.keyfact_aligment(summary_list, keyfact_extra)
        completeness_score, conciseness_score = self.compute_score(batch_pred_alignment_labels, batch_pred_sentence_line_numbers, batch_sentences)
        return completeness_score, conciseness_score

    def test(self, batch_size=3):
        """ Test """
        doc_list = [
            # 'This is a very good idea. Although simple, but very insightful.',
            # 'Can I take a look?',
            # 'Do not trust him, he is a liar.',
        ]
        summary_list = [
            # 'This is a very good idea. Although simple, but very insightful.',
            # 'Can I take a look?',
            # 'Do not trust him, he is a liar.',
        ]
        keyfact=None
        completeness_score, conciseness_score = self.keyfact_to_score(doc_list, summary_list, keyfact)
        print(completeness_score, conciseness_score)

        return None

Testclass = LLamaScorer(device='cuda:0', max_length=6000, checkpoint='chatgpt')

'''----------------------------------------'''
# Load dataset
# File path to your JSON file
csv_file_path = ''

# Check if CSV file exists
if os.path.exists(csv_file_path):
    # Load from CSV if it exists
    processed_data_df = pd.read_csv(csv_file_path)
    print("Loaded data from existing CSV file.")



def extract_values(row):
    document_list = [row['document']]*3
    summary_keys = ["Epoch1_Summary_MoreComple", "Epoch1_Summary_MoreConcise", "Epoch1_Summary_Balance"]
    summary_list = [row[key] for key in summary_keys]
    keyfact_list = [ast.literal_eval(row['extracted_keyfacts'])]*3#Keyfacts

    results = {}

    if [] in summary_list:
        return {}, {}, {}

    try:
        if np.nan in summary_list:
            summary_list = ['placeholder' if isinstance(x, float) and np.isnan(x) else x for x in summary_list]
        score_list = Testclass.keyfact_to_score(document_list, summary_list, keyfact_list)
        results["Epoch1_score_MoreComple"] = {'Completeness': score_list[0][0], 'Conciseness': score_list[1][0]}
        results["Epoch1_score_MoreConcise"] = {'Completeness': score_list[0][1], 'Conciseness': score_list[1][1]}
        results["Epoch1_score_Balance"] = {'Completeness': score_list[0][2], 'Conciseness': score_list[1][2]}
    except Exception as e:
        print(f"Error processing row {row}: {e}")
        return {}, {}, {}

    return (
        results["Epoch1_score_MoreComple"],
        results["Epoch1_score_MoreConcise"],
        results["Epoch1_score_Balance"]
    )

# Apply the function to each row and store the results in a new column
# Take a random sample of 1000 rows
# Filter the DataFrame based on the 'category' column
cut_data_df = processed_data_df
# Check the length
print(len(cut_data_df)) 
cut_data_df[["Epoch1_score_MoreComple", 
                     "Epoch1_score_MoreConcise", 
                     "Epoch1_score_Balance"]] = cut_data_df.progress_apply(extract_values, axis=1).apply(pd.Series)
cut_data_df.to_csv('')
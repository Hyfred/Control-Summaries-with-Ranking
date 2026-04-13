
from transformers import AutoTokenizer
import pandas as pd
import os
import torch
# from unsloth.chat_templates import get_chat_template
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import ast
import transformers
import json 
import re

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
tqdm.pandas()

class LLaMAModel:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint=''):
        # Set up model
        self.device = device
        # self.max_length = max_length
        print('max_length: ', max_length)

        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,#"gabbage/lora_model",
            )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        # FastLanguageModel.for_inference(self.model)
        self.pipeline = transformers.pipeline(
            "text-generation",#"summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},#, "max_seq_length": max_length},
            device_map="auto",
        )

    def inference(self, message):
        outputs = self.pipeline(
            message,
            max_new_tokens=5000,
            # max_length=1000
        )

        summarys_str = [self.extract_json_summary(sum_str[0]['generated_text'][1]['content']) for sum_str in outputs]
        return summarys_str

    def control_inference(self, message):
        outputs = self.pipeline(
            message,
            # max_new_tokens=50000,
            # max_length=1000
        )

        summarys_str = [self.extract_json_summary(sum_str[0]['generated_text'][1]['content']) for sum_str in outputs]
        return summarys_str

    def fix_and_parse_json(self, json_str):
        """Handle JSON strings with unescaped quotes using regex-based repair"""
        try:
            # First try standard JSON parsing
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If that fails, attempt regex-based repair
            match = re.search(r'^\s*{\s*"(?P<key>[^"]+)":\s*"(?P<value>.+?)"\s*}\s*$', 
                            json_str, re.DOTALL)
            if match:
                key = match.group('key')
                value = match.group('value')
                # Escape unescaped quotes while preserving correct ones
                value = re.sub(r'(?<!\\)"', r'\"', value)
                return {key: value}
            raise
    
    def clean_test(self, output):
        start_idx = output.find('{')
        end_idx = output.find('}')
        # sometime llm would produce ) or ] instead of }
        if end_idx == -1:  # If '}' is not found
            if ']' in output:
                output = output[::-1].replace(']', '}', 1)[::-1]  # Replace the last occurrence of ']'
            elif ')' in output:
                output = output[::-1].replace(')', '}', 1)[::-1]  # Replace the last occurrence of ')'
            
            end_idx = output.find('}')  # Try finding '}' again

        output = output[start_idx:end_idx+1]
        output = output.replace('\n','')
        return output

    def extract_json_summary(self, model_output):

        output = model_output.replace('```', '')
        try:                
            output = self.clean_test(output)
            output = ast.literal_eval(output)
            first_key = next(iter(output))
            summary = output[first_key]
            summary = summary.replace('\n', '').replace('[', '').replace(']', '')
            return summary
        except Exception:
            pass
        
        # try to avoid //" escaped quotes issue
        try:
            output = self.clean_test(output)
            parsed = self.fix_and_parse_json(output)
            summary = parsed[next(iter(parsed))]
            summary = summary.replace('\\"', '"').replace("\\'", "'")
            summary = summary.replace('\"', '"').replace("\'", "'")
            return summary.replace('\n', '').translate(str.maketrans('', '', '[]'))
        except Exception as e:
            print(f'JSON parsing failed: {model_output}')
            return ''

    def input_chat_template(self, document, summary=None):
        instruction_morecomple = '''Please summarize the input document, prioritizing completeness over conciseness. Return the summary in the following JSON format: {"Summary": "answer"}'''
        instruction_moreconcise = '''Please summarize the input document, prioritizing conciseness over completeness. Return the summary in the following JSON format: {"Summary": "answer"}'''
        instruction_balance = '''Please summarize the input document, balancing completeness with conciseness. Return the summary in the following JSON format: {"Summary": "answer"}'''
        if summary == None:
            row_json = [
                [{"role": "user", "content": f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction_morecomple}\n\n### Input:\n{document}\n\n### Response:\n\n"}],
                [{"role": "user", "content": f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction_moreconcise}\n\n### Input:\n{document}\n\n### Response:\n\n"}],
                [{"role": "user", "content": f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction_balance}\n\n### Input:\n{document}\n\n### Response:\n\n"}]
            ]        
        else:
            row_json = [{"role": "user", "content": f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{document}\n\n### Response:\n\n"},{"role": "assistant", "content":summary}]
        return row_json


# checkpoint='DISLab/SummLlama3.1-8B'
# checkpoint='Qwen/Qwen2.5-7B-Instruct'
checkpoint='llama_ours'

llamascore = LLaMAModel(device='cuda:1', max_length=50000, checkpoint=checkpoint)
# llamascore.test()

# Load dataset
# File path to your JSON file
csv_file_path = 'test_original.csv'

# Check if CSV file exists
if os.path.exists(csv_file_path):
    # Load from CSV if it exists
    processed_data_df = pd.read_csv(csv_file_path)
    print("Loaded data from existing CSV file.")

# extract category which short document
processed_data_df["category"] = processed_data_df["doc_id"].apply(lambda x: x.split("-")[0])
filtered_df = processed_data_df[processed_data_df['category'].isin(['dialogsum', 'wikihow', 'cnn'])]

def extract_values(row):
    document_list = [row['document']]
    src_input = []
    for doc in document_list:
        src_input.extend(llamascore.input_chat_template(doc))
    try:
        score_list = llamascore.control_inference(src_input)
        more_complete, more_concise, balanced = score_list
        return more_complete, more_concise, balanced
    except Exception as e:
        print(f"Error processing row {row}: {e}")  # Log the error
        return [],[],[]

# Apply the function to each row and store the results in a new column
# Take a random sample of 1000 rows
# Filter the DataFrame based on the 'category' column
# cut_data_df = processed_data_df[processed_data_df['category'].isin(['cnn', 'dialogsum', 'wikihow'])]
# Check the length
# print(len(cut_data_df)) 
filtered_df[["Epoch1_Summary_MoreComple", 
                    "Epoch1_Summary_MoreConcise", 
                    "Epoch1_Summary_Balance"]] = filtered_df.progress_apply(extract_values, axis=1).apply(pd.Series)
filtered_df.to_csv('test_llama_ours.csv')
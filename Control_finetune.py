
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import torch
from transformers import TrainingArguments
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
import pandas as pd
from datasets import Dataset
import os
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
import numpy as np
import ast  # To safely evaluate string representations of dictionaries
import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"


max_seq_length = 6000 # Supports RoPE Scaling interally, so choose any!
# model_option='DISLab/SummLlama3.1-8B' 
# model_option='Qwen/Qwen2.5-7B-Instruct'
model_option="unsloth/mistral-7b-instruct-v0.3"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_option, 
    # load_in_8bit_fp32_cpu_offload=True,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


number_of_token = []

def format_chat_template(document, summary=None, prompt_type="Balance"):
    # Define different types of instructions
    instruction_morecomple = '''Please summarize the input document, prioritizing completeness over conciseness.'''
    instruction_moreconcise = '''Please summarize the input document, prioritizing conciseness over completeness.'''
    instruction_balance = '''Please summarize the input document, balancing completeness with conciseness.'''

    # Choose the correct instruction based on prompt_type
    if prompt_type == "MoreComple":
        instruction = instruction_morecomple
    elif prompt_type == "MoreConcise":
        instruction = instruction_moreconcise
    else:
        instruction = instruction_balance  # Default to "balance"

    # Create the row_json based on whether a summary is provided
    if summary is None:
        row_json = [
            {"role": "user", "content": f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{document}\n\n### Response:\n\n"}
        ]
    else:
        row_json = [
            {"role": "user", "content": f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{document}\n\n### Response:\n\n"},
            {"role": "assistant", "content": summary}
        ]
    
    return row_json



# Preprocessing function for multi-reference setup
def preprocess_function(examples):
    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    types2idx = {
        "MoreComple": 1,
        "Balance": 0,
        "MoreConcise": -1
    } #need to feed type into model training process

    summary_types = [prompt_type for prompt_type in examples["Control_type"]]
    chat_input = [format_chat_template(example,prompt_type=summary_type) for example, summary_type in zip(examples["Document"],summary_types)]
    inputs = tokenizer.apply_chat_template(chat_input, tokenize=True, return_dict=True, max_length=6000, truncation=True, padding="max_length")

    for i in range(0, -1, -1):  # Adjust range if needed
        summary_key = f"Epoch{i}_Control_Summary"
        
        # Check existence and if the summary is non-empty
        if summary_key in examples and examples[summary_key]:
            examples_with_eos = [
                format_chat_template(docu, per_summary[2:-2] + EOS_TOKEN, prompt_type=summary_type)
                for docu, per_summary, summary_type in zip(examples["Document"], examples[summary_key], summary_types)
            ]
            
            with tokenizer.as_target_tokenizer():
                labels = tokenizer.apply_chat_template(
                    examples_with_eos,
                    return_dict=True,
                    max_length=6000,
                    truncation=True,
                    padding="max_length"
                ).input_ids
            
            inputs[f"control_epoch{i}"] = labels
        else:
            print(f"Missing or empty data for {summary_key}")

    for i in range(4, -1, -1):  # first get epoch score, since stack first stack generate summary
        summary_key = f"Epoch{i}_Summary"
        if examples[summary_key]==[]:print(1) # check
        if summary_key in examples and examples[summary_key] is not None:
            # examples_with_eos_test = [f"{sentence}{EOS_TOKEN}" for sentence in examples[summary_key]]
            examples_with_eos = [
                format_chat_template(docu, per_summary[2:-2] + EOS_TOKEN, prompt_type=summary_type)
                for docu, per_summary, summary_type in zip(examples["Document"], examples[summary_key], summary_types)
            ]
            with tokenizer.as_target_tokenizer():
                labels = tokenizer.apply_chat_template(
                    examples_with_eos, return_dict=True, max_length=6000, truncation=True, padding="max_length"
                ).input_ids
                # Replace padding tokens with -100
                # labels = [[(l if l != tokenizer.pad_token_id else -100) for l in sublist] for sublist in labels]
            inputs[f"generate_epoch{i}"] = labels

    # Tokenize each summary as separate labels
    for i in range(1, 10):  # Assuming there are 7 summaries
        summary_key = f"Summary{i}" 
        if summary_key in examples and examples[summary_key] is not None:
            examples_with_eos = [
                format_chat_template(docu, per_summary[2:-2] + EOS_TOKEN, prompt_type=summary_type)
                for docu, per_summary, summary_type in zip(examples["Document"], examples[summary_key], summary_types)
            ]
            with tokenizer.as_target_tokenizer():
                labels = tokenizer.apply_chat_template(
                    examples_with_eos, return_dict=True, max_length=6000, truncation=True, padding="max_length"
                ).input_ids
                # Replace padding tokens with -100
                # labels = [[(l if l != tokenizer.pad_token_id else -100) for l in sublist] for sublist in labels]
            inputs[f"label{i}"] = labels
    inputs["labels"]=[[types2idx[pertype]] for pertype in summary_types]
    inputs["Completeness"] = examples["Completeness"]
    inputs["Conciseness"] = examples["Conciseness"]
    # inputs["KeyfactId"] = examples["Keyfact_id"]

    return inputs

def stat_token(token_lengths): 
    # Calculate basic statistics for token length -> define the proper max token
    mean = np.mean(token_lengths)
    median = np.median(token_lengths)
    q3 = np.percentile(token_lengths, 75)
    percen95 = np.percentile(token_lengths, 95)
    variance = np.var(token_lengths)
    # Display results
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"third quartile: {q3}")
    print(f"95 percentile: {percen95}")
    print(f"Variance: {variance}")

# Load dataset
# File path to your JSON file
csv_file_path = ''#LLaMA_LLS_finesure.csv

# Check if CSV file exists
if os.path.exists(csv_file_path):
    # Load from CSV if it exists
    processed_data_df = pd.read_csv(csv_file_path)
    print("Loaded data from existing CSV file.")


count_nan = 0
# Function to extract 'Completeness' and 'Conciseness' for all rows
def extract_values(row):
    completeness = []
    conciseness = []

    for i in range(0, -1, -1):  # Adjust range if needed
        key = f"Epoch{i}_Control_score"
        score_str = row.get(key, None)
        if score_str is None:
            continue
        try:
            score = ast.literal_eval(score_str)
            completeness.append(score["Completeness"])
            conciseness.append(score["Conciseness"])
        except Exception as e:
            print(e)
            continue

    for i in range(4, -1, -1):  # first get epoch score, since stack first stack generate summary
        score = row[f"Epoch{i}_score"]  # Access the dictionary
        try:
            score = ast.literal_eval(score)
        except Exception as e: 
            global count_nan
            count_nan+=1
            continue
        completeness.append(score["Completeness"])
        conciseness.append(score["Conciseness"])

    for i in range(1, 10):  # Adjust range based on the number of scores (e.g., 1-10)
        score = row[f"score{i}"]  # Access the dictionary
        score = ast.literal_eval(score)
        completeness.append(score["Completeness"])
        conciseness.append(score["Conciseness"])
    
    return pd.Series([completeness, conciseness])

# **Filtering logic**
def is_valid_score(score):
    # Convert from string to dictionary if needed
    if isinstance(score, str):
        try:
            score = ast.literal_eval(score)  # Safely parse string to dict
        except (ValueError, SyntaxError):
            return False  # If parsing fails, treat it as invalid

    if not isinstance(score, dict):  
        return False  # Ensure it's a dictionary

    return bool(score) and not any(pd.isna(value) for value in score.values())

processed_data_df = processed_data_df[processed_data_df['Epoch0_Summary'].astype(bool)]
# processed_data_df = processed_data_df[processed_data_df['Epoch1_Summary'].astype(bool)]
filtered_df = processed_data_df[processed_data_df['Epoch0_score'].apply(is_valid_score)]
# filtered_df = filtered_df[filtered_df['Epoch1_score'].apply(is_valid_score)]
len(filtered_df)

###############
## summary sort by total score
###############

def sort_summaries(df):
    summary_cols = [col for col in df.columns if col.startswith('Summary')]
    score_cols = [col.replace('Summary', 'score') for col in summary_cols]
    
    for idx, row in df.iterrows():
        entries = []
        for i, summary_col in enumerate(summary_cols):
            score_col = score_cols[i]
            if pd.notna(row[summary_col]) and pd.notna(row[score_col]):
                score_dict = ast.literal_eval(row[score_col])
                avg_score = (score_dict.get('Completeness', 0) + score_dict.get('Conciseness', 0)) / 2
                entries.append((summary_col, row[summary_col], score_col, row[score_col], avg_score))
        
        entries.sort(key=lambda x: x[4], reverse=True)
        
        for i, (summary_col, summary, score_col, score, _) in enumerate(entries):
            df.at[idx, summary_cols[i]] = summary
            df.at[idx, score_cols[i]] = score
    
    return df


# filtered_df = sort_summaries(filtered_df)
###############
## summary sort by shuffle
###############
import random

def shuffle_summaries(df):
    summary_cols = [col for col in df.columns if col.startswith('Summary')]
    score_cols = [col.replace('Summary', 'score') for col in summary_cols]
    
    for idx, row in df.iterrows():
        valid_entries = [(col, row[col], score_cols[i], row[score_cols[i]]) 
                         for i, col in enumerate(summary_cols) if pd.notna(row[col]) and pd.notna(row[score_cols[i]])]
        
        random.shuffle(valid_entries)
        
        for i, (summary_col, summary, score_col, score) in enumerate(valid_entries):
            df.at[idx, summary_cols[i]] = summary
            df.at[idx, score_cols[i]] = score
    
    return df

filtered_df = shuffle_summaries(filtered_df)
################
## end
################
# filtered_df = filtered_df[filtered_df['Control_type'] == 'MoreComple'] # quick verify

filtered_df[["Completeness", "Conciseness"]] = filtered_df.apply(extract_values, axis=1)
# processed_data_df['Keyfact_id'] = processed_data_df.groupby('Doc_id').ngroup() # Create a Keyfact_id for each unique Doc_id
# processed_data_df['Keyfacts'] = processed_data_df['Keyfacts'].apply(ast.literal_eval)# Parse Keyfact column using ast.literal_eval
# keyfact_dict = processed_data_df.drop_duplicates('Keyfact_id').set_index('Keyfact_id')['Keyfacts'].to_dict() # Create a dictionary to store Keyfact_id -> Keyfact mapping
print(count_nan)

dataset = Dataset.from_pandas(filtered_df)

# Shuffle and split into training and test sets
shuffled_dataset = dataset.shuffle(seed=7)
train_test_split = shuffled_dataset.train_test_split(train_size=12000, test_size=100) #1170 150

# Accessing the training and test sets
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Tokenize the dataset
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
# stat_token(number_of_token) # do the stat for finding max length

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)


class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_softmax = nn.LogSoftmax(dim=2)  # For your logit dimension
        self.loss_fn = nn.NLLLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
        self.option = 'marginranking_top1_control'
        self.cs_eval = []

    def log(self, logs, *args, **kwargs):
        # Add a custom key to distinguish training vs. evaluation logs
        logs = {f"train_{k}" if self.model.training else f"eval_{k}": v for k, v in logs.items()}
        super().log(logs)

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
        # NLL_mean = loss.view(batch_size, seq_len).mean(dim=-1) # can't directly use mean since length are not fixed.
        # assert torch.equal(NLL_avg, NLL_mean) # not the same

        return NLL_avg
    
    def compute_average(self, numbers):
        if not numbers:  # Check if the list is empty
            return None  # Return None (or another value to signify no data)
        return sum(numbers) / len(numbers)

        
    def RankingLoss(self, score, summary_score=None, margin=1, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
        ones = torch.ones_like(score)
        loss_func = nn.MarginRankingLoss(0.0)
        TotalLoss = loss_func(score, score, ones)
        # candidate loss
        n = score.size(1)
        if not no_cand:
            for i in range(1, n):
                pos_score = score[:, :-i]
                neg_score = score[:, i:]
                pos_score = pos_score.contiguous().view(-1)
                neg_score = neg_score.contiguous().view(-1)
                ones = torch.ones_like(pos_score)
                loss_func = torch.nn.MarginRankingLoss(margin * i, reduction='sum')
                loss = loss_func(pos_score, neg_score, ones)
                TotalLoss += loss
        if no_gold:
            return TotalLoss
        # gold summary loss
        pos_score = summary_score.unsqueeze(-1).expand_as(score)
        neg_score = score
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones_like(pos_score)
        loss_func = torch.nn.MarginRankingLoss(gold_margin)
        TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
        return TotalLoss

    def CorrCoef(self, LL_score, score=None):
        # turn tensor into numpy array
        LL_score = LL_score.cpu().detach().numpy()
        batch_size = len(LL_score)

        # create order 
        if score == None:        
            spearman_correlations = []
            cosine_similarity = []
            order = np.arange(-1, -10, -1)
            for i in range(batch_size):
                sample = LL_score[i]
                
                # Spearman
                spearman_corr, _ = spearmanr(sample, order)
                spearman_correlations.append(spearman_corr)
                
                # cs
                cos_sim = 1 - cosine(sample, order)  #  1-distance
                cosine_similarity.append(cos_sim)
        else:
            spearman_score1_list = []
            spearman_score2_list = []
            
            for item_id, LL_value in enumerate(LL_score):
                Completeness = score[0][item_id]
                Conciseness = score[1][item_id]

                # Spearman
                spearman_score1, _ = spearmanr(LL_value, Completeness)
                spearman_score2, _ = spearmanr(LL_value, Conciseness)
                spearman_score1_list.append(spearman_score1)
                spearman_score2_list.append(spearman_score2)
                
            return self.compute_average(spearman_score1_list),self.compute_average(spearman_score2_list)
        
        return cosine_similarity, spearman_correlations
    
    def CorrCoef_Real(self, LL_score, score=None, prompttypes=None):
        # turn tensor into numpy array
        LL_score = LL_score.cpu().detach().numpy()

        spearman_list = []
        
        for item_id, (LL_value, prompttype) in enumerate(zip(LL_score, prompttypes)):
            Completeness = score[0][item_id]
            Conciseness = score[1][item_id]

            # Spearman
            if prompttype==1:
                spearman_score, _ = spearmanr(LL_value, Completeness)
            elif prompttype==-1:
                spearman_score, _ = spearmanr(LL_value, Conciseness)
            elif prompttype==0:
                spearman_score, _ = spearmanr(LL_value, Completeness+Conciseness)
            spearman_list.append(spearman_score)
            
        return self.compute_average(spearman_list)


    def custom_sort(self, tensor1, score_tensor):
        '''use score_tensor to sort tensor1 in descending order, it allows batch setting'''
        # Stack tensor1 and tensor2 along a new dimension (dim=2) for sorting
        concatenated = torch.stack([tensor1, score_tensor], dim=2)
        # Sort based on the second column (score_tensor values) in descending order
        sorted_concatenated, indices = torch.sort(concatenated[:, :, 1], descending=True)
        # Reorder the first column (tensor1 values) using the sorted indices
        sorted_tensor1 = torch.gather(concatenated[:, :, 0], 1, indices)
        # Return the reordered tensor1
        return sorted_tensor1


    def create_score_tensor(self, Completeness, Conciseness, operation=lambda x, y: (x + y) / 2):
        """
        Combine completeness and conciseness to obtain one score.
        # Handle skipping indices based on nan_list for specific operations.
        """
        def apply_operation(a, b, operation):
            a = np.array(a)
            b = np.array(b)
            return operation(a, b)
        
            # result = []
            # for i in range(len(a)):
            #     result.append(operation(a[i], b[i]))
            # return np.array(result)

        # Calculate merge_score_label (process all indices without skipping)
        merge_score_label = apply_operation(
            Completeness, 
            Conciseness, 
            operation, 
        )

        return merge_score_label

    def create_score_tensor_plus(self, Completeness, Conciseness, prompttypes):
        """
        Combine Completeness and Conciseness scores using different operations based on prompttypes.

        Args:
        - Completeness (np.ndarray): 2D array of completeness scores.
        - Conciseness (np.ndarray): 2D array of conciseness scores.
        - prompttypes (np.ndarray): 1D array with values 1, 0, or -1.

        Returns:
        - np.ndarray: 2D array of computed scores.
        """

        def operation(x, y, ptype):
            if ptype == 0:
                return (x + y) - 0.1 * np.abs(x - y)
            elif ptype == 1:
                return (x + y) - 0.1 * (y - x)
            elif ptype == -1:
                return (x + y) - 0.1 * (x - y)
            else:
                raise ValueError("Invalid prompt type. Must be 1, 0, or -1.")

        # Ensure prompttypes is broadcastable to the shape of Completeness and Conciseness
        prompttypes = np.array(prompttypes).reshape(-1, 1)

        # Apply element-wise operation
        merge_score_label = np.vectorize(operation)(Completeness, Conciseness, prompttypes)

        return merge_score_label

    def mean_ranking_first_value(self, merge_score_label):
        '''This is for evaluation,
        see what is the ranking of the model generation in the candidate list with respect to finesure score.
        Additionally, it creates a list of first_value - max_value for each subarray.'''
        
        rankings = []
        score_differ = []
        
        for subarray in merge_score_label:
            sorted_indices = np.argsort(subarray)[::-1]  # Get indices for descending order, the lower rank the higher score
            rank = np.where(sorted_indices == 0)[0][0] + 1  # Find the rank of the first element (convert to 1-based index)
            rankings.append(rank)
            
            # Compute first_value - max_value
            generate_score = subarray[0]
            max_score = max(subarray)
            value_differ = max_score - generate_score
            
            # Store just the new value in modified_subarrays
            score_differ.append(value_differ)

        return np.mean(rankings), score_differ

    def ratio_first_value(self, merge_score_label, prompttypes):
        ''' input promptype: 2d array
        return the ratio distance,
        compute the distance between ref and genera.
          - if morecomple(prompttype=1): max-first
          - if moreconcise(prompttype=-1): flip the merge_score_label. change to con/com
          - if balance(prompttype=0): compute the merge_score_label-1, select the smallest, first-smallest
        Additionally, it creates a list of first_value - max_value for each subarray.'''
            

        score_differ = []
        
        for subarray, prompttype in zip(merge_score_label, prompttypes):
            if prompttype == 1:
                # morecomple: compute max(subarray) - first_value
                generate_score = subarray[0]
                max_score = max(subarray)
                value_differ = max_score - generate_score

            elif prompttype == -1:
                # moreconcise: convert each score as 1/score,
                # but if a score is 0, use 10 instead
                # then compute max(new_subarray) - new_subarray[0]
                new_subarray = []
                for x in subarray:
                    if x == 0:
                        new_subarray.append(10)
                    else:
                        new_subarray.append(1 / x)
                generate_score = new_subarray[0]
                max_score = max(new_subarray)
                value_differ = max_score - generate_score

            elif prompttype == 0:
                # balance: subtract 1 from each element, take the absolute value, means the distance with 1
                # then compute first element - smallest in modified subarray, we hope generate more close to 1 (samller)
                modified = [abs(x - 1) for x in subarray]
                first_value = modified[0]
                smallest = min(modified)
                value_differ = first_value - smallest

            score_differ.append(value_differ)

        return score_differ

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # loss_func = nn.CrossEntropyLoss(ignore_index=4, reduction='none')
        device = model.device  # detect device
        outputs = model(inputs.input_ids)#model(**inputs) #
        logits = outputs.get("logits") # [4,514,50264][batch, seq_len, dict_len]
        log_probs = self.log_softmax(logits)

        # print(logits.size())
        control0 = inputs.get("control_epoch0").to(device)  # [4,128]
        generate0 = inputs.get("generate_epoch0").to(device)  # [4,128]
        generate1 = inputs.get("generate_epoch1").to(device)  # [4,128]
        generate2 = inputs.get("generate_epoch2").to(device)  # [4,128]
        generate3 = inputs.get("generate_epoch3").to(device)  # [4,128]
        generate4 = inputs.get("generate_epoch4").to(device)  # [4,128]
        label1 = inputs.get("label1").to(device)  # [4,128]
        label2 = inputs.get("label2").to(device)  # [4,128]
        label3 = inputs.get("label3").to(device)  # [4,128]
        label4 = inputs.get("label4").to(device)  # [4,128]
        label5 = inputs.get("label5").to(device)  # [4,128]
        label6 = inputs.get("label6").to(device)  # [4,128]
        label7 = inputs.get("label7").to(device)  # [4,128]
        label8 = inputs.get("label8").to(device)  # [4,128]
        label9 = inputs.get("label9").to(device)  # [4,128]
        prompttypes = inputs.get("labels").view(-1).cpu().numpy()  # [4,1] 1:com 0:bala -1:concise
        Completeness = inputs.get("Completeness").cpu().numpy()
        Conciseness = inputs.get("Conciseness").cpu().numpy()

        # print(label1.size())
        LL_control0 = -self.NLL_loss(log_probs, control0) # [[batch]]
        # LL_generation0 = -self.NLL_loss(log_probs, generate0) # [[batch]]
        # LL_generation1 = -self.NLL_loss(log_probs, generate1) # [[batch]]
        # LL_generation2 = -self.NLL_loss(log_probs, generate2) # [[batch]]
        # LL_generation3 = -self.NLL_loss(log_probs, generate3) # [[batch]]
        LL_generation4 = -self.NLL_loss(log_probs, generate4) # [[batch]]
        LL_label1 = -self.NLL_loss(log_probs, label1)
        LL_label2 = -self.NLL_loss(log_probs, label2)
        LL_label3 = -self.NLL_loss(log_probs, label3)
        LL_label4 = -self.NLL_loss(log_probs, label4)
        LL_label5 = -self.NLL_loss(log_probs, label5)
        LL_label6 = -self.NLL_loss(log_probs, label6)
        LL_label7 = -self.NLL_loss(log_probs, label7)
        LL_label8 = -self.NLL_loss(log_probs, label8)
        LL_label9 = -self.NLL_loss(log_probs, label9)
        # Stack the tensors along dimension 1 (axis 1) to create pairs
        #stack_tensor = torch.stack((LL_generation2, LL_generation1, LL_generation0, LL_label1, LL_label2,LL_label3,LL_label4,LL_label5,LL_label6,LL_label7,LL_label8,LL_label9), dim=1)
        stack_tensor = torch.stack((LL_label1, LL_label2,LL_label3,LL_label4,LL_label5,LL_label6,LL_label7,LL_label8,LL_label9), dim=1)

        concat_result = self.create_score_tensor_plus(Completeness, Conciseness, prompttypes)# combine two to create score tensor. using various type
        concat_result_label_part = concat_result[:, -9:]

        generate_ranking, score_difference = self.mean_ranking_first_value(concat_result) #the score for generation is 1st position
        score_difference = torch.tensor(score_difference, device=device) # batchsize

        # compute ratio different
        ratio_score = self.create_score_tensor(
            Completeness, 
            Conciseness, 
            lambda x, y: np.where(y != 0, x / y, 10)
        ) # get com/con
        ratio_difference = self.ratio_first_value(ratio_score,prompttypes) #the score for generation is 1st position
        ratio_record = np.mean(ratio_difference) # as smaller as good, we hope 0 or negative value, without penalization
        ratio_difference = torch.tensor(ratio_difference, device=device) # batchsize

        score_tensor = torch.tensor(concat_result_label_part, device=device) # Convert to tensor
        sorted_tensor = self.custom_sort(stack_tensor, score_tensor)
        if self.option=='marginranking':
            loss = self.RankingLoss(sorted_tensor, summary_score=None, margin=0.5, gold_margin=0,
                        gold_weight=1, no_gold=True, no_cand=False)/sorted_tensor.size(0)
        if self.option=='marginranking_top1':
            loss1 = self.RankingLoss(sorted_tensor, summary_score=None, margin=0.5, gold_margin=0,
                        gold_weight=1, no_gold=True, no_cand=False)/sorted_tensor.size(0)
            score_differ = torch.mul(torch.exp(LL_generation4),score_difference)
            loss2 = torch.sum(torch.max(torch.tensor(0.0), score_differ)) /sorted_tensor.size(0)
            loss = loss1+100*loss2
        if self.option=='marginranking_top1_control':
            loss1 = self.RankingLoss(sorted_tensor, summary_score=None, margin=0.5, gold_margin=0,
                        gold_weight=1, no_gold=True, no_cand=False)/sorted_tensor.size(0)
            score_differ = torch.mul(torch.exp(LL_control0),score_difference)
            loss2 = torch.sum(torch.max(torch.tensor(0.0), score_differ)) /sorted_tensor.size(0)

            ratio_differ = torch.mul(torch.exp(LL_control0),ratio_difference)
            loss3 = torch.sum(torch.max(torch.tensor(0.0), ratio_differ)) /sorted_tensor.size(0)
            loss = loss1+1*loss2+1*loss3

        spearman_com, spearman_con = self.CorrCoef(stack_tensor,[Completeness[:, -9:],Conciseness[:, -9:]])
        spearman_real = self.CorrCoef_Real(stack_tensor,[Completeness[:, -9:],Conciseness[:, -9:]],prompttypes)
        if self.model.training:
            wandb.log({"train_loss": round(loss.item(), 2), "generate_ranking": generate_ranking, 
                       "train_spear_com": spearman_com, "train_spear_con": spearman_con, 
                       "ratio_diff":ratio_record,"spearman_real":spearman_real})
        else:
            wandb.log({"eval_loss": round(loss.item(), 2), "generate_ranking": generate_ranking, 
                       "eval_spear_com": spearman_com, "eval_spear_con": spearman_con, 
                       "ratio_diff":ratio_record,"spearman_real":spearman_real})
        

        return (loss, outputs) if return_outputs else loss
    
    def reset_metrics(self):
        '''Clear cosine_similarity metrics'''
        self.cs_eval = []
    def get_metrics(self):
        '''Retrieve cosine_similarity metrics'''
        return self.cs_eval
    

# Define the training arguments, specifying the label names
training_args = TrainingArguments(
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    # warmup_steps = 10,
    # max_steps = 60,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 1,
    output_dir = "outputs",
    optim = "adamw_8bit",
    seed = 3407,
    report_to="wandb",
    do_eval=True,
    label_names=[f"label{i}" for i in range(1, 10)] + ["Completeness", "Conciseness"] + [f"control_epoch{i}" for i in range(1)]
      + [f"generate_epoch{i}" for i in range(5)] + ["labels"]  # Specify label names
)

# Initialize the trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    # compute_metrics=compute_metrics
)

wandb.init(project='ControlSum_threetype_allmodel')
# Train the model
trainer_stats = trainer.train()
print(trainer_stats) 


model.save_pretrained("") # Local saving
tokenizer.save_pretrained("")

from Llama_score import LLaMAScorer
import os
import pandas as pd
import ast
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

checkpoint = 'llama_ours'
# checkpoint='DISLab/SummLlama3.1-8B'
# checkpoint="unsloth/mistral-7b-instruct-v0.3"
# checkpoint='Qwen/Qwen2.5-7B-Instruct'

llamascore = LLaMAScorer(device='cuda:0', max_length=6000, checkpoint=checkpoint)
# llamascore.test()

# Load dataset
# File path to your JSON file
csv_file_path = ''

# Check if CSV file exists
if os.path.exists(csv_file_path):
    # Load from CSV if it exists
    processed_data_df = pd.read_csv(csv_file_path)
    print("Loaded data from existing CSV file.")

def extract_values(row):
    summary_list = []
    for i in range(1, 14):  # Adjust range based on the number of summary columns
        column_name = f"Summary{i}"
        if column_name in row and pd.notna(row[column_name]):  # Check if column exists and is not NaN
            summary_list.append(row[column_name])
    document_list = [row['Document']]*len(summary_list)
    src_input = [llamascore.input_chat_template(doc) for doc in document_list]
    tgt_input = [llamascore.input_chat_template(doc,summ) for doc,summ in zip(document_list,summary_list)]
    try:
        score_list = llamascore.score(src_input, tgt_input, batch_size=2)
        return score_list
    except Exception as e:
        print(f"Error processing row {row}: {e}")  # Log the error
        return []

# Apply the function to each row and store the results in a new column
# Take a random sample of 1000 rows
# Filter the DataFrame based on the 'category' column
# cut_data_df = processed_data_df[processed_data_df['category'].isin(['cnn', 'dialogsum', 'wikihow'])]
# Check the length
# processed_data_df= processed_data_df.iloc[:50] # for testing
print(len(processed_data_df)) 
processed_data_df["LL_Score"] = processed_data_df.progress_apply(extract_values, axis=1)
# processed_data_df.to_csv('./ControlSum_v2/testset_epoch1.csv')


# Lists to collect values
completeness_spearman_list = []
conciseness_spearman_list = []
combine_score_spearman_list = []

# Iterate over each row in the dataframe
for _, row in processed_data_df.iterrows():
    # Convert LL_Score to list
    LL_score = row['LL_Score']
    
    # Get the correct number of valid summaries
    number_valid_summary = row['number_valid_summary_list']

    completeness, conciseness, combine_score = [], [], []
    
    for i in range(1, number_valid_summary + 1):  
        summary_key = f"score{i}"
        if summary_key in row:
            score_dict = ast.literal_eval(row[summary_key])
            completeness.append(score_dict.get('Completeness', 0))  
            conciseness.append(score_dict.get('Conciseness', 0))
            combine_score.append((score_dict.get('Total score', 0)+score_dict.get('Conciseness', 0))/2)

    # Ensure lists have valid data for correlation
    if len(LL_score) == len(completeness):
        # Calculate Spearman correlation
        completeness_spearman, _ = spearmanr(LL_score, completeness)
        conciseness_spearman, _ = spearmanr(LL_score, conciseness)
        combine_score_spearman, _ = spearmanr(LL_score, combine_score)

        # Store Spearman correlation values as features for each row
        completeness_spearman_list.append(completeness_spearman)
        conciseness_spearman_list.append(conciseness_spearman)
        combine_score_spearman_list.append(combine_score_spearman)

# Add these lists as new columns in df_full_data
processed_data_df['Comple_Spea'] = completeness_spearman_list
processed_data_df['Concise_Spea'] = conciseness_spearman_list
processed_data_df['Total_Spea'] = combine_score_spearman_list

processed_data_df.to_csv('', index=False)

def print_mean_and_count(df, subset_name):
    # Calculate mean for each correlation type
    completeness_mean = df['Comple_Spea'].mean()
    conciseness_mean = df['Concise_Spea'].mean()
    total_score_mean = df['Total_Spea'].mean() 

    # Count the number of valid summaries
    num_valid_summaries = df['number_valid_summary_list'].count()

    # Print the results
    print(f"Mean Values for {subset_name}:")
    print(f"  Completeness Spearman: {completeness_mean:.4f}")
    print(f"  Conciseness Spearman: {conciseness_mean:.4f}")
    print(f"  Total Score Spearman: {total_score_mean:.4f}")
    print(f"  Number of Valid Summaries: {num_valid_summaries}")
    print("\n")

# Print for testset and trainset
print_mean_and_count(processed_data_df, "Testset")


# Create the DataFrame
correlation_df = pd.DataFrame({
    'Completeness': processed_data_df['Comple_Spea'],
    'Conciseness': processed_data_df['Concise_Spea'],
    'Total Score': processed_data_df['Total_Spea']
})

# Melt the DataFrame to long format for seaborn
correlation_df_melted = correlation_df.melt(var_name='Metric', value_name='Spearman Correlation')

# Set up the plot
plt.figure(figsize=(10, 6))

# Create the boxplot
sns.boxplot(data=correlation_df_melted, x='Metric', y='Spearman Correlation')

# Add a title and labels
plt.title('Box Plot of Correlations: Completeness, Conciseness, and Total Score')
plt.tight_layout()

# Save the figure
plt.savefig("rank_llama_our2.png", dpi=300, bbox_inches='tight')  # Save as PNG with high resolution

# Show the plot
plt.show()
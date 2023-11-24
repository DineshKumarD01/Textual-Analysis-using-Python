#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
# Specify the path to the input Excel file
input_file_path = "Input.xlsx"

# Check if the file exists
if os.path.exists(input_file_path):
    # Load the Excel file
    df = pd.read_excel(input_file_path)
    # Process the data as needed
    # ...
else:
    print("Input file not found:", input_file_path)
output_dir = 'output_text_files'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    
    response = requests.get(url)
    
    soup = BeautifulSoup(response.content,'html.parser')
    
    article_title=soup.title.get_text()
    
    article_text = ""
    paragraphs = soup.find_all('p')
    for paragraph in paragraphs[:-3]:
        article_text += paragraph.get_text() + "\n"
        
    article_text.replace('\n', " ")
    
    text_file_path = os.path.join(output_dir, f'{url_id}.txt')
    with open(text_file_path, 'w', encoding='utf-8') as text_file:
        text_file.write(f"{article_title}\n\n")
        text_file.write(f"\n{article_text}\n\n")
        
    print(f"Extracted and saved content for URL ID {url_id}.")
    
print("Extraction and saving completed.")
        import os

# Folder containing the text files
folder_path = "extracted_files"

# Lines to delete
lines_to_delete = [
    "Automate the Data Management Process",
    "Realtime Kibana Dashboard for a financial tech firm",
    "Data Management, ETL, and Data Automation",
    "Data Management – EGEAS",
    "How To Secure (SSL) Nginx with Let’s Encrypt on Ubuntu (Cloud VM, GCP, AWS, Azure, Linode) and Add Domain",
    "Deploy and view React app(Nextjs) on cloud VM such as GCP, AWS, Azure, Linode",
    "Deploy Nodejs app on a cloud VM such as GCP, AWS, Azure, Linode",
    "Grafana Dashboard – Oscar Awards",
    "Rising IT cities and its impact on the economy, environment, infrastructure, and city life by the year 2040.",
    "Rising IT Cities and Their Impact on the Economy, Environment, Infrastructure, and City Life in Future",
    "Internet Demand’s Evolution, Communication Impact, and 2035’s Alternative Pathways",
    "Rise of Cybercrime and its Effect in upcoming Future",
    "AI/ML and Predictive Modeling",
    "Solution for Contact Centre Problems",
    "How to Setup Custom Domain for Google App Engine Application?",
    "Code Review Checklist"
]

# List the files in the folder
file_names = os.listdir(folder_path)

# Iterate through each file
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)

    # Read the file content
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Filter out the lines to delete
    filtered_lines = [line for line in lines if line.strip() not in lines_to_delete]

    # Write the modified content back to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(filtered_lines)

print("Lines deleted from all files.")
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import osnltk.download('stopwords')
nltk.download('punkt')os.path.abspath('stopwords')

# List of stopwords file paths
stopwords_files = ['StopWords_Names.txt','StopWords_Geographic.txt','StopWords_GenericLong.txt','StopWords_Generic.txt','StopWords_DatesandNumbers.txt', 'StopWords_Currencies.txt','StopWords_Auditor.txt']

# Initialize an empty list to store combined stopwords
combined_stopwords = []

# Read stopwords from each file and combine them
for file_path in stopwords_files:
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        stopwords = file.read().splitlines()
        combined_stopwords.extend(stopwords)

# Remove duplicates by converting the list to a set and back to a list
#combined_stopwords = list(set(combined_stopwords))

# Path to the new combined stopwords file
combined_stopwords_file = "combined_stopwords.txt"

# Write the combined stopwords to the new file
with open(combined_stopwords_file, 'w', encoding='utf-8') as file:
    for word in combined_stopwords:
        file.write(word + '\n')

print("Combined stopwords have been written to", combined_stopwords_file)
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load stopwords
with open('combined_stopwords.txt', 'r') as file:
    stop_words = {line.strip() for line in file}

# Directory paths
input_directory = "extracted_files"
output_directory = "cleaned_files"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process each text file
for filename in os.listdir(input_directory):
    if filename.endswith(".txt"):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        with open(input_path, 'r', encoding='utf-8') as input_file:
            text = input_file.read()
            words = word_tokenize(text)
            filtered_words = [word for word in words if word.lower() not in stop_words]
            processed_text = ' '.join(filtered_words)

            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(processed_text)

print("Stopwords removed from all text files.")
import os
import string

# Directory paths
input_directory = "cleaned_files"
output_directory = "all_cleaned_files"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


# Punctuation characters
#punctuations = string.punctuation
#punctuations_to_remove ="_ - , ' "

# Process each text file
for filename in os.listdir(input_directory):
    if filename.endswith(".txt"):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        with open(input_path, 'r', encoding='utf-8') as input_file:
            text = input_file.read()
            # Remove punctuations
            # text_without_punctuations = ''.join([char for char in text if char not in punctuations and punctuations_to_remove])
            translator = str.maketrans('', '', string.punctuation)
            text_without_punctuations = text.translate(translator)
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(text_without_punctuations)

print("Punctuations removed from all text files.")
import os

# Load positive and negative words from their respective text files
def load_words(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        words = set(file.read().split())
    return words

positive_words = load_words('positive_words.txt')
negative_words = load_words('negative_words.txt')

# Directory path for the 100 text files
text_files_directory = "all_cleaned_files"

# Initialize a dictionary to store positive and negative words for each text file
text_file_word_dicts = {}

# Process each text file
for filename in os.listdir(text_files_directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(text_files_directory, filename)

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            words = text.split()

            # Identify positive and negative words
            positive_words_found = [word for word in words if word in positive_words]
            negative_words_found = [word for word in words if word in negative_words]

            # Create a dictionary for the current text file
            word_dict = {
                'positive': positive_words_found,
                'negative': negative_words_found
            }

            # Add the dictionary to the higher-level dictionary
            text_file_word_dicts[filename] = word_dict

# Print the dictionaries for each text file
for filename, word_dict in text_file_word_dicts.items():
    print("File:", filename)
    print("Positive Words:", word_dict['positive'])
    print("Negative Words:", word_dict['negative'])
    print("=" * 50)
def calculate_scores(positive_words, negative_words):
    positive_score = sum(1 for word in positive_words)
    negative_score = sum(-1 for word in negative_words) * -1
    return positive_score, negative_score

higher_level_dictionary = text_file_word_dicts

scores_per_dictionary = []

for dictionary_name, dictionary in higher_level_dictionary.items():
    positive_words = dictionary["positive"]
    negative_words = dictionary["negative"]
    
    positive_score, negative_score = calculate_scores(positive_words, negative_words)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    
    scores_per_dictionary.append((dictionary_name, positive_score, negative_score, polarity_score))

for idx, (dictionary_name, positive_score, negative_score, polarity_score) in enumerate(scores_per_dictionary, start=1):
    print(f"Dictionary {idx} ({dictionary_name}):")
    print(f"Positive Score: {positive_score}")
    print(f"Negative Score: {negative_score}")
    print(f"Polarity Score: {polarity_score}")
    print()
import os

def count_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        words = content.split()
        return len(words)

folder_path = 'all_cleaned_files'  # Replace this with the actual path to your folder

file_word_counts = {}  # Dictionary to store file names and their word counts

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):  # Only consider text files
        file_path = os.path.join(folder_path, filename)
        word_count = count_words(file_path)
        file_word_counts[filename] = word_count

for filename, word_count in file_word_counts.items():
    print(f"File: {filename}, Word Count: {word_count}")
# Sample data
scores_list = scores_per_dictionary

word_count_dict = file_word_counts

# Merging the data
merged_data = {}

for entry in scores_list:
    name, positive_score, negative_score, polarity_score = entry

    if name in word_count_dict:
        word_count = word_count_dict[name]

        entry_data = {
            "positive_score": positive_score,
            "negative_score": negative_score,
            "polarity_score": polarity_score,
            "word_count": word_count
        }

        merged_data[name] = entry_data

# Print the merged data
import json
print(json.dumps(merged_data, indent=4))final_data = merged_data

for file_name, data in final_data.items():
    positive_score = data["positive_score"]
    negative_score = data["negative_score"]
    word_count = data["word_count"]
    
    subjectivity_score = (positive_score + negative_score) / (word_count + 0.000001)
    
    data["subjectivity_score"] = subjectivity_score

# Print the updated file_data
import json
print(json.dumps(final_data, indent=4))

from textstat import textstatimport os
import re
from textstat import textstat

# Replace 'path_to_text_files' with the actual path to your text files
path_to_text_files = 'extracted_files'
output_path = 'analysis2_files'

def calculate_fog_index(text):
    return textstat.gunning_fog(text)

def custom_syllable_count(word):
    if word.endswith(('es', 'ed')):
        return 0
    
    vowels = 'aeiouAEIOU'
    count = 0
    in_vowel_group = False
    for char in word:
        if char in vowels:
            if not in_vowel_group:
                count += 1
                in_vowel_group = True
        else:
            in_vowel_group = False
    return count

def calculate_metrics(text):
    fog_index = calculate_fog_index(text)
    avg_sentence_length = textstat.avg_sentence_length(text)
    
    words = re.findall(r'\b\w+\b', text)
    
    complex_word_count = 0
    for word in words:
        if custom_syllable_count(word) >= 2:  # Custom condition for complex words
            complex_word_count += 1
    
    total_word_count = len(words)
    percentage_complex_words = (complex_word_count / total_word_count) * 100
    
    syllable_count = sum(custom_syllable_count(word) for word in words)
    
    personal_pronouns = ["I", "we", "my", "ours", "us"]
    personal_pronoun_count = sum(1 for word in words if word.lower() in personal_pronouns)
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    return fog_index, avg_sentence_length, percentage_complex_words, complex_word_count, syllable_count, personal_pronoun_count, avg_word_length

def process_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        metrics = calculate_metrics(text)
        return metrics

def main():
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(path_to_text_files):
        if filename.endswith('.txt'):
            file_path = os.path.join(path_to_text_files, filename)
            metrics = process_text_file(file_path)
            
            output_filename = os.path.splitext(filename)[0] + '.txt'
            output_file_path = os.path.join(output_path, output_filename)
            
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(f"Fog Index: {metrics[0]:.2f}\n")
                output_file.write(f"Average Sentence Length: {metrics[1]:.2f}\n")
                output_file.write(f"Percentage of Complex Words: {metrics[2]:.2f}\n")
                output_file.write(f"Complex Word Count: {metrics[3]}\n")
                output_file.write(f"Syllable Count: {metrics[4]}\n")
                output_file.write(f"Personal Pronoun Count: {metrics[5]}\n")
                output_file.write(f"Average Word Length: {metrics[6]:.2f}\n")

if __name__ == '__main__':
    main()


import os

# Replace 'data' with your dictionary of dictionaries
data = final_data

# Replace 'output_folder' with the path where you want to save the text files
output_folder = 'analysis1_files'

def save_dictionary_to_text_file(file_name, inner_dict):
    file_path = os.path.join(output_folder, f"{file_name}.txt")
    with open(file_path, 'w') as file:
        for key, value in inner_dict.items():
            file.write(f"{key}: {value}\n")

def main():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name, inner_dict in data.items():
        save_dictionary_to_text_file(file_name, inner_dict)

if __name__ == '__main__':
    main()

# In[36]:


#sssssssssssssssssssssssssssss

import os

def combine_files(input_folder1, input_folder2, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the list of files in both input folders
    files1 = os.listdir(input_folder1)
    files2 = os.listdir(input_folder2)

    # Combine the files from both folders
    for file1, file2 in zip(files1, files2):
        with open(os.path.join(input_folder1, file1), 'r') as f1, \
             open(os.path.join(input_folder2, file2), 'r') as f2:
            combined_text = f1.read() + f2.read()

        output_file_path = os.path.join(output_folder, f"combined_{file1}")
        with open(output_file_path, 'w') as output_file:
            output_file.write(combined_text)

if __name__ == "__main__":
    input_folder1 = "analysis_files1"
    input_folder2 = "analysis_files2"
    output_folder = "analysis_files"

    combine_files(input_folder1, input_folder2, output_folder)
import pandas as pddf=pd.read_excel('Output Data Structure.xlsx')df.tail()import pandas as pd
import os
import re

directory = 'analysing_files'

def extract_numbers(text):
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', text)
    return [float(num) for num in numbers]

metrics = {
    'positive_score': [],
    'negative_score': [],
    'polarity_score': [],
    'word_count': [],
    'subjectivity_score': [],
    'Fog Index': [],
    'Average Sentence Length': [],
    'Percentage of Complex Words': [],
    'Complex Word Count': [],
    'Syllable Count': [],
    'Personal Pronoun Count': [],
    'Average Word Length': []
}

file_names = os.listdir(directory)

for file_name in file_names:
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'r') as file:
        content = file.read()
        numbers = extract_numbers(content)
        for metric, value in zip(metrics.keys(), numbers):
            metrics[metric].append(value)

df1 = pd.DataFrame(metrics)

df1import pandas as pd

# Example dataframes

# Extract the first two columns of df1
df1_first_two = df[['URL_ID', 'URL']]

# Merge df2 with df1_first_two
merged_df = pd.concat([df1_first_two, df1], axis=1)

print(merged_df)
merged_dfimport os
import re
import pandas as pd

directory = 'extracting_files'

def syllable_count(word):
    word = word.lower()
    if word.endswith(('es', 'ed')):
        return 0
    vowels = 'aeiouy'
    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if count == 0:
        count = 1
    return count

def average_words_per_sentence(text):
    sentences = re.split(r'[.!?]', text)
    words_per_sentence = [len(re.findall(r'\b\w+\b', sentence)) for sentence in sentences]
    return sum(words_per_sentence) / len(words_per_sentence)

syllable_counts = []
avg_words_per_sentence = []

file_names = os.listdir(directory)

for file_name in file_names:
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'r', encoding = 'utf-8') as file:
        content = file.read()
        
        words = re.findall(r'\b\w+\b', content)
        filtered_words = [word for word in words if syllable_count(word) > 2]
        total_syllables = sum(syllable_count(word) for word in filtered_words)
        total_words = len(filtered_words)
        
        avg_words = average_words_per_sentence(content)
        
        syllable_counts.append(total_syllables / total_words)
        avg_words_per_sentence.append(avg_words)

data = {
    'Syllables per Word': syllable_counts,
    'Average Words per Sentence': avg_words_per_sentence
}

result_df = pd.DataFrame(data)


result_df# Merge df2 with df1_first_two
merged1_df = pd.concat([merged_df, result_df], axis=1)

print(merged1_df)merged1_dfdel merged1_df['Syllable Count']merged1_dfdesired_order = ['URL_ID', 'URL', 'positive_score', 'negative_score', 'polarity_score', 'subjectivity_score', 'Average Sentence Length', 'Percentage of Complex Words', 'Fog Index', 'Average Words per Sentence', 'Complex Word Count', 'word_count', 'Syllables per Word', 'Personal Pronoun Count', 'Average Word Length']
merged1_df = merged1_df[desired_order]merged1_dfnew_names={'positive_score':'POSITIVE SCORE', 'negative_score':'NEGATIVE SCORE', 'polarity_score':'POLARITY SCORE', 'subjectivity_score':'SUBJECTIVITY SCORE', 'Average Sentence Length':'AVG SENTENCE LENGTH', 'Percentage of Complex Words':'PERCENTAGE OF COMPLEX WORDS', 'Fog Index':'FOG INDEX', 'Average Words per Sentence':'AVERAGE NUMBER OF WORDS PER SENTENCE', 'Complex Word Count':'COMPLEX WORD COUNT', 'word_count':'WORD COUNT', 'Syllables per Word':'SYLLABLE PER WORD', 'Personal Pronoun Count':'PERSONAL PRONOUNS', 'Average Word Length':'AVG WORD LENGTH'}
merged1_df1 = merged1_df.rename(columns=new_names)merged1_df1output_file = 'Output Data Structures.xlsx'  # Name of the output Excel file

merged_dataframe = merged1_df1.to_excel(output_file, index=False)  # Save the DataFrame to the Excel file
# In[ ]:





# In[ ]:





# In[ ]:





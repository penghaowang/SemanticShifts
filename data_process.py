import pandas as pd
from tqdm import tqdm
import concurrent.futures
import os
from pathlib import Path

def convert_to_sentences(df, test_limit=100, num_workers=4, use_process_pool=True):
    '''
    Convert DataFrame to a list of sentences
    
    Args:
        df: DataFrame
        test_limit: Number of sentences to process, -1 means process all sentences
        num_workers: Number of worker threads/processes for parallel processing
        use_process_pool: Whether to use ProcessPoolExecutor instead of ThreadPoolExecutor, suitable for CPU-bound tasks
    
    Returns:
        sentences: List of sentences
        sentence_info: List of sentence information
    '''
    # Get unique PACO:SENTENCEID values
    unique_sentence_ids = df['PACO:SENTENCEID'].unique()

    if test_limit == -1:
        test_limit = len(unique_sentence_ids)  # If test_limit is -1, process the entire set

    sentence_ids_to_process = unique_sentence_ids[:test_limit]

    def process_sentence(sentence_id):
        group = df[df['PACO:SENTENCEID'] == sentence_id].sort_values('ID')
        words = group['FORM'].astype(str).tolist()
        
        sentence = ''
        for i, word in enumerate(words):
            if i > 0 and str(group['MISC'].iloc[i-1]) != 'SpaceAfter=No':
                sentence += ' '
            sentence += word
        
        return {
            'PACO:SENTENCEID': sentence_id,
            'PACO:TEXTID': group['PACO:TEXTID'].iloc[0],
            'sentence': sentence
        }

    sentences = []
    sentence_info = []

    # Select parallel processing method
    executor_class = concurrent.futures.ProcessPoolExecutor if use_process_pool else concurrent.futures.ThreadPoolExecutor
    
    with executor_class(max_workers=num_workers) as executor:
        futures = [executor.submit(process_sentence, sentence_id) for sentence_id in sentence_ids_to_process]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(sentence_ids_to_process), desc="Processing sentences"):
            result = future.result()
            sentences.append(result['sentence'])
            sentence_info.append(result)

    return sentences, sentence_info

def load_and_process_text_data(relation_path):
    '''Load and process text relation data'''
    text_df = pd.read_csv(relation_path, sep='\t', comment='#')
    text_df.columns = ['PACO:TEXTID', 'MISC']

    new_row = pd.DataFrame({'PACO:TEXTID': [2000000], 'MISC': ['SourceFile=bulletin_ocr_1970_1_en']})
    text_df = pd.concat([new_row, text_df], ignore_index=True)

    text_df['Year'] = text_df['MISC'].str.extract(r'(\d{4})')
    
    text_df['PACO:TEXTID'] = text_df['PACO:TEXTID'].astype(int)
    
    return text_df

def load_data(file_path, columns):
    '''Load TSV data file'''
    try:
        return pd.read_csv(file_path, sep='\t', skiprows=1, names=columns, quoting=3)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print(f"File path: {file_path}")
        return None

def process_and_save_data(data_path, relation_path, output_path, test_limit=-1, num_workers=8):
    '''Main function to process and save data'''
    # Define column names
    columns = ["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC", "PACO:TOKENID", "PACO:SENTENCEID", "PACO:TEXTID"]
    
    # Load data
    df = load_data(data_path, columns)
    if df is None:
        return False
    
    # Process sentences
    sentences, sentence_info = convert_to_sentences(df, test_limit=test_limit, num_workers=num_workers, use_process_pool=True)
    
    # Load and process text data
    text_df = load_and_process_text_data(relation_path)
    
    # Create sentence DataFrame
    sentence_df = pd.DataFrame(sentence_info)
    sentence_df['PACO:TEXTID'] = sentence_df['PACO:TEXTID'].astype(int)
    
    # Merge data
    merged_df = pd.merge(sentence_df, text_df, on='PACO:TEXTID', how='left')
    merged_df = merged_df.sort_values('PACO:SENTENCEID')
    
    # Select required columns
    merged_df = merged_df[['sentence', 'Year', 'MISC', 'PACO:TEXTID']]
    
    # Save as CSV file
    merged_df.to_csv(output_path, index=False)
    
    # Print year range
    min_year = merged_df['Year'].min()
    max_year = merged_df['Year'].max()
    print(f"Year range: from {min_year} to {max_year}")
    print(f"Data processing complete, saved as {output_path}")
    
    return True

if __name__ == "__main__":
    demo = False
    
    # Data path configuration
    data_configs = {
        'news': {
            'data_path': 'downloaded_files/PaCoCo/Credit_Suisse/CS_News/cs_news.token.en.tsv/cs_news.token.en.tsv',
            'relation_path': 'downloaded_files/PaCoCo/Credit_Suisse/CS_News/cs_news.text.en.tsv/cs_news.text.en.tsv',
            'output_path': 'cs_news_en.csv'
        },
        'bulletin_ocr': {
            'data_path': "downloaded_files/PaCoCo/Credit_Suisse/CS_Bulletin_OCR/cs_bulletin_ocr.token.en.ts/cs_bulletin_ocr.token.en.tsv",
            'relation_path': 'downloaded_files/PaCoCo/Credit_Suisse/CS_Bulletin_OCR/cs_bulletin_ocr.text.en.tsv/cs_bulletin_ocr.text.en.tsv',
            'output_path': 'cs_bulletin_ocr_en.csv'
        },
        'bulletin_pdf': {
            'data_path': 'downloaded_files/PaCoCo/Credit_Suisse/CS_Bulletin_PDF/cs_bulletin_pdf.token.en.tsv/cs_bulletin_pdf.token.en.tsv',
            'relation_path': 'downloaded_files/PaCoCo/Credit_Suisse/CS_Bulletin_PDF/cs_bulletin_pdf.text.en.tsv/cs_bulletin_pdf.text.en.tsv',
            'output_path': 'cs_bulletin_pdf_en.csv'
        }
    }
    
    # Select dataset to process
    dataset = 'news'
    config = data_configs[dataset]
    
    # Set processing limit
    test_limit = 100 if demo else -1
    
    # Process data
    process_and_save_data(
        config['data_path'], 
        config['relation_path'], 
        config['output_path'], 
        test_limit=test_limit, 
        num_workers=8
    )

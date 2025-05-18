#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collect semantic shift data for all words, including JSD and entropy values
Read from generated CSV files and merge into summary CSV files
"""

import os
import pandas as pd
import glob
from pathlib import Path
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Any
from logger_config import setup_logger

# Set up logging
logger = setup_logger('collect_semantic_data', 'logs/collect_semantic_data.log')

def load_and_combine_csv_files(file_pattern: str, description: str = "Data") -> Optional[pd.DataFrame]:
    """
    Load and combine all CSV files matching the specified pattern
    
    Args:
        file_pattern: File path pattern using glob format
        description: Data type description for logging
        
    Returns:
        Combined DataFrame, returns None if no files are found
    """
    files = glob.glob(file_pattern)
    if not files:
        logger.warning(f"No CSV files found matching {file_pattern}")
        return None
    
    dataframes = []
    for file in files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            logger.info(f"Read {file}")
        except Exception as e:
            logger.error(f"Error reading {file}: {str(e)}")
    
    if not dataframes:
        logger.error(f"No valid {description} files found")
        return None
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined {len(combined_df)} rows of {description}")
    
    return combined_df

def process_word_directory(args: Tuple[str, str, str, str]) -> Dict[str, Any]:
    """
    Process a single word directory, load its JSD and entropy data
    
    Args:
        args: Tuple containing (word_dir, base_dir, time_bin_level, full_path)
        
    Returns:
        Dictionary containing all data for the word
    """
    word_dir, base_dir, time_bin_level, full_path = args
    result = {"word": word_dir}
    
    # Find JSD CSV file for the word
    jsd_pattern = os.path.join(full_path, f"jsd_changes_{time_bin_level}_*.csv")
    df_jsd = load_and_combine_csv_files(jsd_pattern, f"JSD data for word '{word_dir}'")
    result["jsd"] = df_jsd
    
    # Find entropy CSV file for the word
    entropy_pattern = os.path.join(full_path, f"entropy_changes_{time_bin_level}_*.csv")
    df_entropy = load_and_combine_csv_files(entropy_pattern, f"Entropy data for word '{word_dir}'")
    result["entropy"] = df_entropy
    
    # Find meaning entropy CSV file for the word
    meaning_pattern = os.path.join(full_path, f"meaning_entropy_changes_{time_bin_level}_*.csv")
    df_meaning = load_and_combine_csv_files(meaning_pattern, f"Meaning entropy data for word '{word_dir}'")
    result["meaning_entropy"] = df_meaning
    
    return result

def collect_all_data(base_dir: str = 'semantic_shift_plots', time_bin_level: str = 'period') -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Collect JSD and entropy data for all words in parallel
    
    Args:
        base_dir: Base directory containing all word directories
        time_bin_level: Time binning level ('period' or 'year')
    
    Returns:
        Tuple (combined_jsd, combined_entropy, combined_meaning_entropy)
    """
    # Get all word directories
    word_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    logger.info(f"Found {len(word_dirs)} word directories")
    
    all_jsd_data = []
    all_entropy_data = []
    all_meaning_entropy_data = []
    
    # Prepare arguments for parallel processing
    process_args = [(word_dir, base_dir, time_bin_level, os.path.join(base_dir, word_dir)) for word_dir in word_dirs]
    
    # Process all word directories in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_word_directory, process_args))
    
    # Organize results
    for result in results:
        if result["jsd"] is not None:
            all_jsd_data.append(result["jsd"])
        if result["entropy"] is not None:
            all_entropy_data.append(result["entropy"])
        if result["meaning_entropy"] is not None:
            all_meaning_entropy_data.append(result["meaning_entropy"])
    
    # Combine JSD data for all words
    combined_jsd = None
    if all_jsd_data:
        combined_jsd = pd.concat(all_jsd_data, ignore_index=True)
        logger.info(f"Total combined {len(combined_jsd)} rows of JSD data")
    else:
        logger.error("No JSD data files found")
    
    # Combine entropy data for all words
    combined_entropy = None
    if all_entropy_data:
        combined_entropy = pd.concat(all_entropy_data, ignore_index=True)
        logger.info(f"Total combined {len(combined_entropy)} rows of entropy data")
    else:
        logger.error("No entropy data files found")
    
    # Combine meaning entropy data for all words
    combined_meaning_entropy = None
    if all_meaning_entropy_data:
        combined_meaning_entropy = pd.concat(all_meaning_entropy_data, ignore_index=True)
        logger.info(f"Total combined {len(combined_meaning_entropy)} rows of meaning entropy data")
    
    return combined_jsd, combined_entropy, combined_meaning_entropy

def save_data_and_create_pivot(df: pd.DataFrame, output_dir: str, file_prefix: str, 
                              time_bin_level: str, pivot_params: Dict[str, Any]) -> None:
    """
    Save data and create a pivot table
    
    Args:
        df: DataFrame to save
        output_dir: Output directory
        file_prefix: File name prefix
        time_bin_level: Time binning level
        pivot_params: Parameters for creating the pivot table
    """
    # Save combined data
    output_file = os.path.join(output_dir, f"{file_prefix}_{time_bin_level}.csv")
    df.to_csv(output_file, index=False)
    logger.info(f"Data saved to {output_file}")
    
    # Create pivot table
    try:
        pivot_df = pd.pivot_table(df, **pivot_params)
        pivot_output = os.path.join(output_dir, f"{file_prefix}_by_word_time_{time_bin_level}.csv")
        pivot_df.to_csv(pivot_output)
        logger.info(f"Pivot table created, saved to {pivot_output}")
    except Exception as e:
        logger.error(f"Error creating pivot table: {str(e)}")

def main():
    """Main function"""
    base_dir = 'semantic_shift_plots'
    time_bin_level = 'period'  # Optional 'year' or 'period'
    output_dir = 'semantic_shift_summary'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all data in parallel
    combined_jsd, combined_entropy, combined_meaning_entropy = collect_all_data(base_dir, time_bin_level)
    
    # Process JSD data
    if combined_jsd is not None:
        save_data_and_create_pivot(
            combined_jsd, 
            output_dir, 
            'all_words_jsd',
            time_bin_level,
            {
                'values': 'js_div',
                'index': ['word', 'pos'],
                'columns': 'time_pair',
                'aggfunc': 'mean'
            }
        )
    
    # Process entropy data
    if combined_entropy is not None:
        save_data_and_create_pivot(
            combined_entropy, 
            output_dir, 
            'all_words_entropy',
            time_bin_level,
            {
                'values': 'avg_entropy',
                'index': ['word', 'pos'],
                'columns': 'time_bin',
                'aggfunc': 'mean'
            }
        )
    
    # Process meaning entropy data
    if combined_meaning_entropy is not None:
        output_file = os.path.join(output_dir, f"all_words_meaning_entropy_{time_bin_level}.csv")
        combined_meaning_entropy.to_csv(output_file, index=False)
        logger.info(f"All meaning entropy data saved to {output_file}")

if __name__ == "__main__":
    main()
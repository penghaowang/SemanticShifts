import pandas as pd
from tqdm import tqdm
import concurrent.futures
import os
from pathlib import Path

def convert_to_sentences(df, test_limit=100, num_workers=4, use_process_pool=True):
    '''
    将数据框转换为句子列表
    
    参数:
        df: 数据框
        test_limit: 处理句子的数量，-1 表示处理所有句子
        num_workers: 并行处理的工作线程/进程数
        use_process_pool: 是否使用进程池而非线程池，适用于CPU密集型任务
    
    返回:
        sentences: 句子列表
        sentence_info: 句子信息列表
    '''
    # 获取唯一的 PACO:SENTENCEID 值
    unique_sentence_ids = df['PACO:SENTENCEID'].unique()

    if test_limit == -1:
        test_limit = len(unique_sentence_ids)  # 如果 test_limit 为 -1，则处理全集

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

    # 选择并行处理方式
    executor_class = concurrent.futures.ProcessPoolExecutor if use_process_pool else concurrent.futures.ThreadPoolExecutor
    
    with executor_class(max_workers=num_workers) as executor:
        futures = [executor.submit(process_sentence, sentence_id) for sentence_id in sentence_ids_to_process]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(sentence_ids_to_process), desc="处理句子"):
            result = future.result()
            sentences.append(result['sentence'])
            sentence_info.append(result)

    return sentences, sentence_info

def load_and_process_text_data(relation_path):
    '''加载并处理文本关系数据'''
    text_df = pd.read_csv(relation_path, sep='\t', comment='#')
    text_df.columns = ['PACO:TEXTID', 'MISC']

    new_row = pd.DataFrame({'PACO:TEXTID': [2000000], 'MISC': ['SourceFile=bulletin_ocr_1970_1_en']})
    text_df = pd.concat([new_row, text_df], ignore_index=True)

    text_df['Year'] = text_df['MISC'].str.extract(r'(\d{4})')
    
    text_df['PACO:TEXTID'] = text_df['PACO:TEXTID'].astype(int)
    
    return text_df

def load_data(file_path, columns):
    '''加载TSV数据文件'''
    try:
        return pd.read_csv(file_path, sep='\t', skiprows=1, names=columns, quoting=3)
    except Exception as e:
        print(f"加载数据时出错：{str(e)}")
        print(f"文件路径：{file_path}")
        return None

def process_and_save_data(data_path, relation_path, output_path, test_limit=-1, num_workers=8):
    '''处理并保存数据的主函数'''
    # 定义列名
    columns = ["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC", "PACO:TOKENID", "PACO:SENTENCEID", "PACO:TEXTID"]
    
    # 加载数据
    df = load_data(data_path, columns)
    if df is None:
        return False
    
    # 处理句子
    sentences, sentence_info = convert_to_sentences(df, test_limit=test_limit, num_workers=num_workers, use_process_pool=True)
    
    # 加载并处理文本数据
    text_df = load_and_process_text_data(relation_path)
    
    # 创建句子数据框
    sentence_df = pd.DataFrame(sentence_info)
    sentence_df['PACO:TEXTID'] = sentence_df['PACO:TEXTID'].astype(int)
    
    # 合并数据
    merged_df = pd.merge(sentence_df, text_df, on='PACO:TEXTID', how='left')
    merged_df = merged_df.sort_values('PACO:SENTENCEID')
    
    # 选择所需列
    merged_df = merged_df[['sentence', 'Year', 'MISC', 'PACO:TEXTID']]
    
    # 保存为CSV文件
    merged_df.to_csv(output_path, index=False)
    
    # 打印年份范围
    min_year = merged_df['Year'].min()
    max_year = merged_df['Year'].max()
    print(f"年份范围：从 {min_year} 到 {max_year}")
    print(f"数据处理完成，已保存为 {output_path}")
    
    return True

if __name__ == "__main__":
    demo = False
    
    # 数据路径配置
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
    
    # 选择要处理的数据集
    dataset = 'news'
    config = data_configs[dataset]
    
    # 设置处理限制
    test_limit = 100 if demo else -1
    
    # 处理数据
    process_and_save_data(
        config['data_path'], 
        config['relation_path'], 
        config['output_path'], 
        test_limit=test_limit, 
        num_workers=8
    )

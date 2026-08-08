import numpy as np
import pandas as pd
import pickle
import random
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertModel
import torch

def sliding_window_pooling(text, tokenizer, model, max_len=512, stride=256, verbose=False):
   
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=None,
        truncation=False,
        return_tensors=None,
    )
    input_ids = encoding['input_ids']
    total_tokens = len(input_ids)
    
    if total_tokens <= max_len:
        inputs = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = model(**inputs)
        vec = outputs.pooler_output.squeeze(0).cpu().numpy().tolist()
        return vec, 1

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    tokens = input_ids[1:-1]
    total_len = len(tokens)
    
    segment_vectors = []
    segment_indices = []
    
    for start in range(0, total_len, stride):
        end = min(start + max_len - 2, total_len)
        segment_tokens = tokens[start:end]
        segment_ids = [cls_id] + segment_tokens + [sep_id]
        
        segment_indices.append((start, end))
        
        padding_len = max_len - len(segment_ids)
        if padding_len > 0:
            segment_ids = segment_ids + [tokenizer.pad_token_id] * padding_len
        attention_mask = [1] * (len(segment_ids) - padding_len) + [0] * padding_len
        
        input_tensor = torch.tensor([segment_ids])
        mask_tensor = torch.tensor([attention_mask])
        
        with torch.no_grad():
            outputs = model(input_ids=input_tensor, attention_mask=mask_tensor)
        cls_vec = outputs.pooler_output.squeeze(0).cpu().numpy()
        segment_vectors.append(cls_vec)
        
        if end == total_len:
            break


    avg_vec = np.mean(segment_vectors, axis=0)
    return avg_vec.tolist(), len(segment_vectors)


def get_text_feature(text, tokenizer, model):
    vec, seg_num = sliding_window_pooling(text, tokenizer, model, verbose=False)
    return vec, seg_num


def format_data(vecs):
    return [','.join(str(x) for x in vec) for vec in vecs]


if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('./mc-bert-pretrain')
    model = BertModel.from_pretrained('./mc-bert-pretrain')
    model.eval()

    train_df = pd.read_excel('./data/text1/train.xlsx')
    
    train_vecs = []
    train_seg_counts = []
    for idx, text in tqdm(enumerate(train_df['text']), total=len(train_df)):
        text = str(text)
        vec, seg_num = get_text_feature(text, tokenizer, model)
        train_vecs.append(vec)
        train_seg_counts.append(seg_num)
    
    train_df['segments_count'] = train_seg_counts
    train_df['768vec'] = format_data(train_vecs)
    
    pca = PCA(n_components=8)
    train_vecs_8d = pca.fit_transform(np.array(train_vecs))
    
    train_df['8vec'] = format_data(train_vecs_8d)
    train_df.to_excel("./data/train_with_features.xlsx", index=False)
    
    with open("./data/pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)

    test_df = pd.read_excel('./data/text1/test.xlsx')
    
    test_vecs = []
    test_seg_counts = []
    for idx, text in tqdm(enumerate(test_df['text']), total=len(test_df)):
        text = str(text)
        vec, seg_num = get_text_feature(text, tokenizer, model)
        test_vecs.append(vec)
        test_seg_counts.append(seg_num)
    
    test_df['segments_count'] = test_seg_counts
    test_df['768vec'] = format_data(test_vecs)
    
    test_vecs_8d = pca.transform(np.array(test_vecs))
    
    test_df['8vec'] = format_data(test_vecs_8d)
    test_df.to_excel("./data/test_with_features.xlsx", index=False)
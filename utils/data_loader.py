import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import re

class TextPreprocessor:
    def __init__(self, max_vocab_size=10000, max_len=100):
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        
    def clean_text(self, text):
        """Basic text cleaning"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        word_freq = Counter()
        for text in texts:
            words = self.clean_text(text).split()
            word_freq.update(words)
        
        # Get most common words
        most_common = word_freq.most_common(self.max_vocab_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        words = self.clean_text(text).split()
        sequence = [self.word2idx.get(word, 1) for word in words]  # 1 is <UNK>
        
        # Pad or truncate
        if len(sequence) < self.max_len:
            sequence = sequence + [0] * (self.max_len - len(sequence))
        else:
            sequence = sequence[:self.max_len]
        
        return sequence


def load_spam_dataset(file_path='data/spam.csv'):
    """
    Load and preprocess spam dataset
    Expected format: CSV with 'text' and 'label' columns
    """

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                label, text = parts
                data.append((label, text))
    df = pd.DataFrame(data, columns=['label', 'text'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # df = pd.read_csv(file_path, sep='\t')
    
    # # Convert labels to binary (assuming 'spam' and 'ham' labels)
    # df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        df['text'].values, df['label'].values, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Build vocabulary and preprocess
    preprocessor = TextPreprocessor()
    preprocessor.build_vocab(X_train)
    
    X_train_seq = [preprocessor.text_to_sequence(text) for text in X_train]
    X_val_seq = [preprocessor.text_to_sequence(text) for text in X_val]
    X_test_seq = [preprocessor.text_to_sequence(text) for text in X_test]
    
    return {
        'X_train': np.array(X_train_seq),
        'y_train': y_train,
        'X_val': np.array(X_val_seq),
        'y_val': y_val,
        'X_test': np.array(X_test_seq),
        'y_test': y_test,
        'preprocessor': preprocessor
    }
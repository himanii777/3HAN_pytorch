import pandas as pd
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
import csv

"""
get_vocab_size: # of unique vocabs in our dataset, if its a huge number so we might not be using that number for our embedding dimension. Your embedding dimension can be made way less that that according to you num_classes.
In theory: embedding_dimensions =  number_of_categories**0.25
"""
def get_vocab_size(fake_news_path, true_news_path, text_column='text', add_unknown_token=True):
    fake_news_df = pd.read_csv(fake_news_path , nrows=1500)
    true_news_df = pd.read_csv(true_news_path , nrows=1500)
    all_texts = fake_news_df[text_column].tolist() + true_news_df[text_column].tolist()
    tokenized_texts = [text.split() for text in all_texts]
    all_tokens = [token for sublist in tokenized_texts for token in sublist]
    vocab_counter = Counter(all_tokens)
    vocab = {word: idx for idx, (word, _) in enumerate(vocab_counter.items(), start=1)}
    if add_unknown_token:
        vocab["<UNK>"] = 0
    vocab_size = len(vocab)
    return vocab_size

def build_vocab(fake_news_path, true_news_path, max_vocab_size=20000, min_freq=5):  
    fake_news = pd.read_csv(fake_news_path, nrows=1500)
    true_news = pd.read_csv(true_news_path, nrows=1500)
    data = pd.concat([fake_news, true_news], ignore_index=True)

    # Tokenize the text
    all_text = ' '.join(data['text'])
    tokens = word_tokenize(all_text.lower())

    # Count the frequency of each word
    word_freq = Counter(tokens)

    # Filter words by minimum frequency and limit to max_vocab_size
    vocab = {word: idx for idx, (word, freq) in enumerate(word_freq.items()) if freq >= min_freq}
    if len(vocab) > max_vocab_size:
        # If the filtered vocab is still too large, sort by frequency and take the top `max_vocab_size` items
        vocab = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True)[:max_vocab_size])
    else:
        # Sort remaining words by frequency for consistent indexing
        vocab = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))

    # Add special tokens to the vocabulary
    vocab['<PAD>'] = len(vocab)
    vocab['<UNK>'] = len(vocab) + 1
    return vocab


"""
We need to pad the words according to the max length to have consistent data
"""

def get_max_lengths(data_path):
    word_length_list = []
    sent_length_list = []
    max_data_points = 1500 #change this acc to your data

    with open(data_path, newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for i, row in enumerate(reader):
            if i >= max_data_points:
                break  
            text = row['text'].lower()
            sent_list = sent_tokenize(text)
            sent_length_list.append(len(sent_list))
            for sent in sent_list:
                word_list = word_tokenize(sent)
                word_length_list.append(len(word_list))
        sorted_word_length = sorted(word_length_list)
        sorted_sent_length = sorted(sent_length_list)

    # We will return the 80th percentile for word length and sentence length
    return sorted_word_length[int(0.8 * len(sorted_word_length))], sorted_sent_length[int(0.8 * len(sorted_sent_length))]

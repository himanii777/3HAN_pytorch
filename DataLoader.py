import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from nltk.tokenize import sent_tokenize, word_tokenize

class FakeNewsDataset(Dataset):
    def __init__(self, fake_news_path, true_news_path, max_sentences, max_words_per_sentence, vocab):
        # I limited the data due to memory issues, the csv has almost 20,000 datasets
        self.fake_news = pd.read_csv(fake_news_path, nrows=1500)
        self.true_news = pd.read_csv(true_news_path, nrows=1500)
        self.fake_news['label'] = 0  # 0 for fake news
        self.true_news['label'] = 1  # 1 for true news
        self.data = pd.concat([self.fake_news, self.true_news], ignore_index=True)
        # Shuffle the data
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.max_sentences = max_sentences
        self.max_words_per_sentence = max_words_per_sentence
        self.vocab = vocab  # Vocabulary mapping 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Change this as per your datasets
        text = self.data.iloc[idx]["text"]
        label = self.data.iloc[idx]["label"]

        # Tokenize
        sentences = sent_tokenize(text)[:self.max_sentences]  # split text into sentences
        words = [word_tokenize(sentence)[:self.max_words_per_sentence] for sentence in sentences]

        # Pad sentences and words according to max words and max sentences.
        padded_words = np.full((self.max_sentences, self.max_words_per_sentence), self.vocab['<PAD>'], dtype=int)
        for i, sentence in enumerate(words):
            for j, word in enumerate(sentence):
                padded_words[i, j] = self.vocab.get(word, self.vocab['<UNK>'])  # Use <UNK> for unknown words
        return torch.tensor(padded_words, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def get_dataset(fake_news_path, true_news_path, max_sentences, max_words_per_sentence, batch_size, vocab):
    dataset = FakeNewsDataset(fake_news_path, true_news_path, max_sentences, max_words_per_sentence, vocab)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Datashape = (batch_size, max_sentences, max_words_per_sentence)

        
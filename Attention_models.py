import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import get_vocab_size

"""
From the paper:
WordLevel Attention -> SentenceLevel Attention -> TitleLevel Attention -> Final news vector
The WordAttention processes each sentence and returns a context vector for each sentence.
Then we stack the context vectors and create tensors of shape: (batch_size, num_sentences, hidden_size * num_directions).
This stacked tensor is the input to the SentenceAttention, which processes the sentence representations to generate a document representation. The document representations are then passed to the TitleAttention to generate the final representation.
"""

class WordAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, bidirectional=True, dropout=0.5):
        super(WordAttention, self).__init__()

        
        """
        embed size: embedding dim for each words
        hidden_size: # hidden features
        vocab_size: typically refers to # of unique words in a sentence but the dimensions will be huge otherwise, so, we use a smaller number where embedding itself manages by replacing less frequent words with a special token like <UNK> (unknown).
        num_layers: # of recurrent layers in GRU
        we only use dropouts for num_layers >1
        """   
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_size)
        dim = hidden_size * 2 if bidirectional else hidden_size
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        self.mlp = nn.Linear(dim, dim)
        self.attn_score = nn.Linear(dim, 1, bias=False)

    def forward(self, x, hidden_state):
        embedding = self.embedding(x)  # (batch_size, max_sentence_len, embed_size)
        gru, hidden_state_out = self.gru(embedding, hidden_state)  # (batch_size, max_sentence_len, hidden_size * num_directions), (num_directions*num_layers, batch_size, hidden_size) not sure of hidden-state

        # Hidden representation u_i
        u_i = torch.tanh(self.mlp(gru))  # (batch_size, max_sentence_len, hidden_size * num_directions)

        # Attention scores
        attn_scores = self.attn_score(u_i).squeeze(-1)  # (batch_size, max_sentence_len)

        # Attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, max_sentence_length)

        attn_weights = attn_weights.unsqueeze(-1)  # (batch_size, max_sentence_len, 1)
        context_vector = torch.sum(attn_weights * gru, dim=1)  # (batch_size, hidden_size * num_directions)

        return context_vector, hidden_state_out


class SentenceAttention(nn.Module):
    def __init__(self, hidden_size, word_hidden_size, num_layers=1, bidirectional=True, dropout=0.5):
        super(SentenceAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.word_hidden_size = word_hidden_size
        dim = hidden_size * 2 if bidirectional else hidden_size
        self.gru = nn.GRU(word_hidden_size * 2, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        self.mlp = nn.Linear(dim, dim)
        self.attn_score = nn.Linear(dim, 1, bias=False)

    def forward(self, x, hidden_state):
        
        gru, hidden_state = self.gru(x, hidden_state)  # (batch_size, num_sentences, hidden_size * num_directions)
        u_i = torch.tanh(self.mlp(gru))  # (batch_size, num_sentences, hidden_size * num_directions)
        attn_scores = self.attn_score(u_i).squeeze(-1)  # (batch_size, num_sentences)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, num_sentences)
        attn_weights = attn_weights.unsqueeze(-1)  # (batch_size, num_sentences, 1)
        context_vector = torch.sum(attn_weights * gru, dim=1)  # (batch_size, hidden_size * num_directions)
        return context_vector, hidden_state


    
class TitleAttention(nn.Module):
    def __init__(self, hidden_size, sent_hidden_size, num_classes, num_layers=1, bidirectional=True, dropout=0.5):
        super(TitleAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        dim = hidden_size * 2 if bidirectional else hidden_size
        self.gru = nn.GRU(sent_hidden_size * 2, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        self.mlp = nn.Linear(dim, dim)
        self.attn_score = nn.Linear(dim, 1, bias=False)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x, hidden_state):
        gru, hidden_state = self.gru(x, hidden_state)  # (batch_size, num_titles, hidden_size * num_directions)
        u_i = torch.tanh(self.mlp(gru))  # (batch_size, num_titles, hidden_size * num_directions)
        attn_scores = self.attn_score(u_i).squeeze(-1)  # (batch_size, num_titles)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, num_titles)
        attn_weights = attn_weights.unsqueeze(-1)  # (batch_size, num_titles, 1)
        context_vector = torch.sum(attn_weights * gru, dim=1)  # (batch_size, hidden_size * num_directions)
        output = self.fc(context_vector)  # (batch_size, num_classes)
        return output, hidden_state



#Debug code

# fake_news_path = r"----"
# true_news_path= r"----"
# vocab_size = get_vocab_size(fake_news_path, true_news_path)
# word_attention = WordAttention(embed_size=100, hidden_size=128, vocab_size=vocab_size,  num_layers=1)
# sentence_attention = SentenceAttention(hidden_size=128, num_layers=1)
# title_attention = TitleAttention(hidden_size=128, num_layers=1)
# sample_input = torch.randint(0, vocab_size, (32, 10, 20))  # (batch_size, num_sentences, max_sentence_length)
# sentence_context_vectors = []
# for i in range(sample_input.size(1)):
#     sentence = sample_input[:, i, :]  # (batch_size, max_sentence_length)
#     context_vector, _ = word_attention(sentence)
#     sentence_context_vectors.append(context_vector)
# # Stack context vectors to form sentence representations
# sentence_context_vectors = torch.stack(sentence_context_vectors, dim=1)  # (batch_size, num_sentences, hidden_size * num_directions)

# document_vector, _ = sentence_attention(sentence_context_vectors)

# title_input = document_vector.unsqueeze(1)  # (batch_size, 1, hidden_size * num_directions)
# final_vector, _ = title_attention(title_input)

# print("Word vector shape:", context_vector.shape)
# print("Sentence vector shape:", document_vector.shape)
# print("Title vector shape:", final_vector.shape)



import torch
from Attention_models import *
from Utils import*
import torch
import torch.nn as nn
import torch.nn.functional as F

class HAN_3_model(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, title_hidden_size, embed_size, vocab_size, batch_size, num_classes, max_sent_length, max_word_length):
        super(HAN_3_model, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.title_hidden_size = title_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.num_classes = num_classes

        #layers
        self.word_att_net = WordAttention(embed_size, word_hidden_size, vocab_size, num_layers=1)
        self.sent_att_net = SentenceAttention(sent_hidden_size, word_hidden_size, num_layers=1)
        self.title_att_net = TitleAttention(title_hidden_size, sent_hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        batch_size = input.size(0)
        num_sentences = input.size(1)

        sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        title_hidden_state = torch.zeros(2, batch_size, self.title_hidden_size).cuda()

        output_list = []
        input = input.permute(1, 0, 2)  # (num_sentences, batch_size, max_sentence_length)
        for i in input:
            word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size).cuda()
            output, word_hidden_state = self.word_att_net(i, word_hidden_state)
            output_list.append(output)
        
        output = torch.stack(output_list, dim=1)  # (batch_size, num_sentences, hidden_size * num_directions)
        output, sent_hidden_state = self.sent_att_net(output, sent_hidden_state)  # (batch_size, hidden_size * num_directions)

        output = output.unsqueeze(1)  # Add a dummy dimension for the title level (batch_size, 1, hidden_size * num_directions)
        output, title_hidden_state = self.title_att_net(output, title_hidden_state)  # (batch_size, num_classes)
        output= self.sigmoid(output)

        return output




# Debug code
# fake_news_path = r"--------"
# true_news_path = r"--------"
# vocab_size = get_vocab_size(fake_news_path, true_news_path)

# hier_att_net = HAN_3_model(word_hidden_size=128, sent_hidden_size=128, title_hidden_size=128, embed_size=100, vocab_size=vocab_size, batch_size=32, num_classes=2, max_sent_length=10, max_word_length=20)

# sample_input = torch.randint(0, vocab_size, (32, 10, 20))  # (batch_size, num_sentences, max_sentence_length)
# output = hier_att_net(sample_input)
# print("Output shape:", output.shape)







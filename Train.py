from Utils import*
from DataLoader import*
from Attention_models import*
from HAN_3_model import*
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.optim.lr_scheduler import StepLR


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def train(opt):
    model = HAN_3_model(opt.word_hidden_size, opt.sent_hidden_size, opt.title_hidden_size, 
                        opt.embed_size, opt.vocab_size, opt.batch_size, opt.num_classes,
                        opt.max_sentence_length, opt.max_word_length)
    print("-----------Model Initialized-----------")
    vocab = build_vocab(opt.fake_news_path, opt.true_news_path)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params}")

    train_loader, val_loader = get_dataset(opt.fake_news_path, opt.true_news_path, 
                                           opt.max_sentence_length, opt.max_word_length, 
                                           opt.batch_size, vocab)
    print("Finished Loading The Data")
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)  # LR decaying after certain epochs edit this for training
    criterion = opt.criterion
    model.train()
    print("-----------Training Started-------------")

    #compile the losses
    losses = []
    all_labels = []
    all_preds = []
    for epoch in range(opt.num_epochs):
        total_loss = 0
        for features, labels in train_loader:
            features = features.long().cuda() #change it to long
            labels = labels.long().cuda()
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, preds = torch.max(predictions, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        scheduler.step()
        average_loss = total_loss / len(train_loader)
        losses.append(average_loss)
        print(f"Epoch: {epoch+1}/{opt.num_epochs}, Loss: {average_loss:.4f}")

        #save the model if necessary
        # if (epoch+1) % 100 == 0:
        #     torch.save(model.state_dict(), f'3HAN_epoch_{epoch+1}.pt')

    # loss graph
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

class Options():
    fake_news_path=r"C:\Users\Himani\Desktop\Main Files\DL projects\3HAN\3HAN\Fake.csv"
    true_news_path=r"C:\Users\Himani\Desktop\Main Files\DL projects\3HAN\3HAN\True.csv"
    max_word_fake, max_sent_fake =get_max_lengths(fake_news_path) #max words, max sentence
    max_word_true, max_sent_true=get_max_lengths(true_news_path)
    max_sentence_length= max(max_sent_fake, max_sent_true)
    max_word_length= max(max_word_fake, max_sent_fake)  
    vocab_size=get_vocab_size(fake_news_path, true_news_path)
    criterion= nn.CrossEntropyLoss()
    word_hidden_size=50
    sent_hidden_size=50
    title_hidden_size=50
    batch_size=32
    lr=0.01 
    num_epochs=100
    num_classes=2
    embed_size=512


if __name__ == "__main__":
    opt = Options()
    train(opt)

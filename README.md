# 3HAN_pytorch

This repository contains 3HAN: A Hierarchical Attention Network for Fake News Detection. Official Paper link: https://link.springer.com/chapter/10.1007/978-3-319-70096-0_59

# Model Introduction
<img width="535" alt="image" src="https://github.com/user-attachments/assets/3a2a6e62-dbde-4b7d-a110-5c1af3ca1a8c">

3HAN utilizes three attention levels, focusing on words, sentences, and the headline, to construct a news vector: an effective representation of an input news article. It processes the article in a hierarchical bottom-up manner. Since the headline is a distinguishing feature of fake news and only a few words and sentences in an article carry more significance than others, 3HAN assigns differential importance to various parts of the article through its three layers of attention. Basically: Word Level Attention -> Sentence Level Attention -> Title Level Attention -> Final News Vector. BidirectionalGRU is used for sequence encoding.
![image](https://github.com/user-attachments/assets/291a03a0-e136-4d91-9834-21377c30ef0a)

# Datasets used
I used the following datasets from Kaggle: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

# Train Instructions

If you want to change the parameters, check out Train.py-> Class Options where you can also add the path to your data.

To run the code, simply go to the terminal and type:

```python
python Train.py
```
# Results
<img width="386" alt="image" src="https://github.com/user-attachments/assets/fdac8256-b8c9-4e33-8859-c6c02809503c">

Breakdown of the Confusion Matrix
True Positives (TP): 115,682 (bottom right)
True Negatives (TN): 114,152 (top left)
False Positives (FP): 5,348 (top right) 
False Negatives (FN): 4,818 (bottom left) 

Performance Analysis
Accuracy: The overall accuracy of the model is (TP + TN) / Total = (115,682 + 114,152) / (115,682 + 114,152 + 5,348 + 4,818) ≈ 0.957. SO about 95.7% of the predictions were correct.

Precision (True Class):  TP / (TP + FP) = 115,682 / (115,682 + 5,348) ≈ 0.956

Recall (True Class):TP / (TP + FN) = 115,682 / (115,682 + 4,818) ≈ 0.960

There is a good balance between false positives and false negatives, so the model does not heavily favor one class over the other.













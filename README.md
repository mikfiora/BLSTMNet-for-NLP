# BLSTMNet-for-NLP
## BLSTMNET

Michele Fioravanti

repo for the project for Natural Language Processing

---


## DATASET: TREC (Text REtrieval Conference)

The Text REtrieval Conference (TREC) Question Classification dataset contains 5500 labeled questions in training set and another 500 for test set.
TASK: Classification of questions into 6 different labels (coarse labels, 50 fine labels also present but not used)


---


## Slide 3

abbreviation ("What is the full form of .com ?")
entity ("What fowl grabs the spotlight after the Chinese Year of the Monkey ?")
description ("Why do heavier objects travel downhill faster ?")
human being ("Who killed Gandhi ?")
location ("What sprawling U.S. state boasts the most airports ?")
numeric value ("How many points make up a perfect five pin bowling score ?")

LABELS:


---


## 

![Slide 4 Image](/mnt/data/ppt_images/slide_4.png)


---


## 

![Slide 5 Image](/mnt/data/ppt_images/slide_5.png)


---


## other useful data on the dataset:

max length: 33 tokens
average length: 10 tokens
vocabulary length: 8902 tokens
tokens obtained by keras tokenizer


---


## Basic Preliminary Processing

represent labels by vectors of length 6 ([0,1,0,0,0,0] for label “1”)
reduced all letters to lowercase
stripped all sentences of punctuation
every word was added to a vocabulary and replaced by its index
padded every sentence (with zeros) to 33 (max sentence length)


---


## BASELINE MODELS:

simple models created and trained to give a baseline for the actual candidate models:
1 B.O.W. + LR
2 random embeddings + LR 


---


## GloVe: Global Vectors for Word Representation

Both candidate models use this word representation technique.
Open source project by Stanford University.
Based on co-occurrence statistics: we expect the ratio of 2 word vectors to be close to 1 when they have the same co-occurence probability w.r.t the chosen context word.
of the 8902 words in the vocabulary: 8450 found in GloVe


---


## FIRST CANDIDATE MODEL: BLSTM2DCNN

TAKEN FROM (no code provided in the paper):

![Slide 10 Image](/mnt/data/ppt_images/slide_10.png)


---


## BLSTM2DCNN (word embedding size 3, hidden units 5)

![Slide 11 Image](/mnt/data/ppt_images/slide_11.png)


---


## Slide 12

![Slide 12 Image](/mnt/data/ppt_images/slide_12.png)

![Slide 12 Image](/mnt/data/ppt_images/slide_12.png)

BLSTM

LSTM Unit


---


## BLSTM2DCNN: Why?

“RNN can capitalize on distributed representations of words by first converting the tokens comprising each text into vectors, which form a matrix. This matrix includes two dimensions: the time-step dimension and the feature vector dimension, and it will be updated in the process of learning feature representation.”
RNNs ignore feature vector dimension.
CNNs very successful in image processing: output of rnn is a matrix “image”:
“ It is a good choice to utilize 2D convolution and 2D pooling to sample more meaningful features on both the time-step dimension and the feature vector dimension for text classification. “


---


## BLSTM2DCNN

from the paper:
“The dimension of word embeddings is 300, the hidden units of LSTM is 300. We use 100 convolutional filters each for window sizes of (3,3), 2D pooling size of (2,2). We set the mini-batch size as 10 and the learning rate of AdaDelta as the default value 1.0. For regularization, we employ Dropout operation (Hinton et al., 2012) with dropout rate of 0.5 for the word embeddings, 0.2 for the BLSTM layer and 0.4 for the penultimate layer, we also use l2 penalty with coefficient 10−5 over the parameters. “
5,455,606 parameters (including GloVe Embeddings, which are finetuned during training)



---


## SECOND CANDIDATE MODEL: BLSTMNet

Very similar to BLSTM2DCNN, but with added depth:
instead of 1 convolutional layer and 1 max pooling layer, we have 3 for both (alternated).
main advantages: 
-way less parameters (way smaller matrix passed to output layer): 3,355,581 trainable parameters, 61% of BLSTM2DCNN and Just 23% more of the L.R. model.
-added depth to the model, which should better capture features in the matrix outputted by BLSTM.


---


## RESULTS

score: accuracy on test set

![Slide 16 Image](/mnt/data/ppt_images/slide_16.png)

As expected, label 0 the worst in every model due to little data

BLSTMNet outperforms BLSTM2DCNN and many other models like AdaSent

first baseline model too simple to obtain good score

second baseline model overfit the training data, with bigger and bigger overfitting when the size of the word embeddings is increased. (train accuracy:0.978)

BLSTMNet performs better than some models considered by the paper’s authors while maintaining a lower parameter count



---


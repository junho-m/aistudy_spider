# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 22:29:01 2020

@author: 82103
"""


import tensorflow as tf
import numpy as np
import pandas as pd
from IPython.display import Image

paragraph_dict_list = [
         {'paragraph': 'dishplace is located in sunnyvale downtown there is parking around the area but it can be difficult to find during peak business hours my sisters and i came to this place for dinner on a weekday they were really busy so i highly recommended making reservations unless you have the patience to wait', 'category': 'food'},
         {'paragraph': 'service can be slower during busy hours but our waiter was courteous and help gave some great entree recommendations', 'category': 'food'},
         {'paragraph': 'portions are huge both french toast and their various omelettes are really good their french toast is probably 1.5x more than other brunch places great place to visit if you are hungry and dont want to wait 1 hour for a table', 'category': 'food'},
         {'paragraph': 'we started with apps going the chicken and waffle slides and chicken nachos the sliders were amazing and the nachos were good too maybe by themselves the nachos would have scored better but after those sliders they were up against some tough competition', 'category': 'food'},
         {'paragraph': 'the biscuits and gravy was too salty two people in my group had the gravy and all thought it was too salty my hubby ordered a side of double egg and it was served on two small plates who serves eggs to one person on separate plates we commented on that when it was delivered and even the server laughed and said she doesnt know why the kitchen does that presentation of food is important and they really missed on this one', 'category': 'food'},
         {'paragraph': 'the garlic fries were a great starter (and a happy hour special) the pancakes looked and tasted great and were a fairly generous portion', 'category': 'food'},
         {'paragraph': 'our meal was excellent i had the pasta ai formaggi which was so rich i didnt dare eat it all although i certainly wanted to excellent flavors with a great texture contrast between the soft pasta and the crisp bread crumbs too much sauce for me but a wonderful dish', 'category': 'food'},
         {'paragraph': 'what i enjoy most about palo alto is so many restaurants have dog-friendly seating outside i had bookmarked italico from when they first opened about a 1.5 years ago and was jonesing for some pasta so time to finally knock that bookmark off', 'category': 'food'},
         {'paragraph': 'the drinks came out fairly quickly a good two to three minutes after the orders were taken i expected my iced tea to taste a bit more sweet but this was straight up green tea with ice in it not to complain of course but i was pleasantly surprised', 'category': 'food'},
         {'paragraph': 'despite the not so good burger the service was so slow the restaurant wasnt even half full and they took very long from the moment we got seated to the time we left it was almost 2 hours we thought that it would be quick since we ordered as soon as we sat down my coworkers did seem to enjoy their beef burgers for those who eat beef however i will not be returning it is too expensive and extremely slow service', 'category': 'food'},
    
         {'paragraph': 'the four reigning major champions simona halep caroline wozniacki angelique kerber and defending us open champion sloane stephens could make a case for being the quartet most likely to succeed especially as all but stephens has also enjoyed the no1 ranking within the last 14 months as they prepare for their gruelling new york campaigns they currently hold the top four places in the ranks', 'category': 'sports'},
         {'paragraph': 'the briton was seeded nn7 here last year before a slump in form and confidence took her down to no46 after five first-round losses but there have been signs of a turnaround including a victory over a sub-par serena williams in san jose plus wins against jelena ostapenko and victoria azarenka in montreal. konta pulled out of new haven this week with illness but will hope for good things where she first scored wins in a major before her big breakthroughs to the semis in australia and wimbledon', 'category': 'sports'},
         {'paragraph': 'stephens surged her way back from injury in stunning style to win her first major here last year—and ranked just no83 she has since proved what a big time player she is winning the miami title via four fellow major champions then reaching the final at the french open back on north american hard courts she ran to the final in montreal only just edged out by halep she has also avoided many of the big names in her quarter—except for wild card azarenka as a possible in the third round', 'category': 'sports'},
         {'paragraph': 'when it came to england chances in the world cup it would be fair to say that most fans had never been more pessimistic than they were this year after enduring years of truly dismal performances at major tournaments – culminating in the 2014 event where they failed to win any of their three group games and finished in bottom spot those results led to the resignation of manager roy hodgson', 'category': 'sports'},
         {'paragraph': 'the team that eliminated russia – croatia – also improved enormously during the tournament before it began their odds were 33/1 but they played with real flair and star players like luka modric ivan rakitic and ivan perisic showed their quality on the world stage having displayed their potential by winning all three of their group stage games croatia went on to face difficult tests like the semi-final against england', 'category': 'sports'},
         {'paragraph': 'the perseyside outfit finished in fourth place in the premier league table and without a trophy last term after having reached the champions league final before losing to real madrid', 'category': 'sports'},
         {'paragraph': 'liverpool fc will return to premier league action on saturday lunchtime when they travel to leicester city in the top flight as they look to make it four wins in a row in the league', 'category': 'sports'},
         {'paragraph': 'alisson signed for liverpool fc from as roma this summer and the brazilian goalkeeper has helped the reds to keep three clean sheets in their first three premier league games', 'category': 'sports'},
         {'paragraph': 'but the rankings during that run-in to new york hid some very different undercurrents for murray had struggled with a hip injury since the clay swing and had not played a match since losing his quarter-final at wimbledon and he would pull out of the us open just two days before the tournament began—too late however to promote nederer to the no2 seeding', 'category': 'sports'},
         {'paragraph': 'then came the oh-so-familiar djokovic-nadal no-quarter-given battle for dominance in the third set there were exhilarating rallies with both chasing to the net both retrieving what looked like winning shots nadal more than once pulled off a reverse smash and had his chance to seal the tie-break but it was djokovic serving at 10-9 who dragged one decisive error from nadal for a two-sets lead', 'category': 'sports'}
]
df = pd.DataFrame(paragraph_dict_list)
df = df[['paragraph', 'category']]

df.head()

df.tail()

results = set()
df['paragraph'].str.lower().str.split().apply(results.update)

idx2word= dict(enumerate(results))
word2idx = {v: k for k, v in idx2word.items()}

word2idx["bread"]

word2idx["tournament"]

def encode_paragraph(paragraph):
    words = paragraph.split(" ")
    encoded = []
    for word in words:
        encoded.append([word2idx[word]])
    
    return encoded

df["enc_paragraph"] = df.paragraph.apply(encode_paragraph)

def encode_category(category):
    if category == "food":
        return [1,0]
    else:
        return [0,1]
    
df["enc_category"] = df.category.apply(encode_category)

def word_cnt(paragraph):
    return len(paragraph.split(" "))

df["seq_length"] = df.paragraph.apply(word_cnt)

max_word_cnt = 0
for row in df['paragraph']:
    if len(row.split(" ")) > max_word_cnt:
        max_word_cnt = len(row.split(" "))

print (max_word_cnt)

def sequence_padding(enc_paragraph):
    seq_length = len(enc_paragraph)
    for i in range(seq_length,max_word_cnt):
        enc_paragraph.append([-1])
    
    return enc_paragraph

df["enc_paragraph"] = df.enc_paragraph.apply(sequence_padding)

print(df["enc_paragraph"][0])

df.head(20)

enc_paragraph = np.array(df.enc_paragraph.tolist())
enc_category = np.array(df.enc_category.tolist())
seq_length = np.array(df.seq_length.tolist())

train_X = enc_paragraph
train_Y = enc_category
train_X.shape
train_Y.shape
Image(url= "https://raw.githubusercontent.com/captainchargers/deeplearning/master/img/lstm_model_overview.png", width=500, height=250)

tf.reset_default_graph()
tf.set_random_seed(777)

# hyperparameters
learning_rate = 0.001
n_epochs = 300

# define input
X = tf.placeholder(tf.float32, [None,max_word_cnt,1])
# define output
y = tf.placeholder(tf.int32, [None, 2])

# word embedding layer
embedding = tf.layers.dense(X, 5)
# LSTM cell
cell = tf.nn.rnn_cell.LSTMCell(num_units=64)
# get output and state (we only use state in this example)
output, state = tf.nn.dynamic_rnn(
    cell, embedding, 
    dtype=tf.float32, sequence_length=seq_length)
# state is contextual vector which is input to MLP classifier (dense layer) below
dense_layer = tf.layers.dense(state.h, 32)
# logit has two nodes for cross entropy with one hot encoding target
logits = tf.layers.dense(dense_layer, 2)
# loss function
cross_entropy = \
    tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y, logits=logits)
loss = tf.reduce_mean(cross_entropy)
# Adam Optimizor
optimizer = \
    tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(loss)
    
print(X)
print(embedding)
print(state)
print(dense_layer)
print(logits)

# initialize the variables
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for epoch in range(1, n_epochs+1):
        sess.run(optimizer, 
                 feed_dict={X: train_X, y: train_Y})
        train_loss = sess.run(loss, 
                              feed_dict={X: train_X, 
                                         y: train_Y})

        if epoch == 1 or epoch % 50 == 0:
            preds = tf.nn.softmax(logits)  
            correct_prediction = \
                tf.equal(tf.argmax(preds, 1), 
                         tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(
                        tf.cast(
                            correct_prediction, "float"))
            cur_acc = accuracy.eval(
                {X: train_X, y: train_Y})
            print("epoch: "+ str(epoch)+", loss: "+
                  str(train_loss)+", acc: "+str(cur_acc))
            
            

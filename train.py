import pandas as pd
import numpy as np
import re
import csv
import tensorflow as tf
import nltk
from gensim.models import Word2Vec
from keras.preprocessing import text, sequence

df = pd.read_csv('train.csv')
raw_input = df['comment_text']

sentences = []
for num in range(len(raw_input)):
    temp = nltk.sent_tokenize(raw_input[num])
    for j in range(len(temp)):
        txt = re.sub('[^\w\s\']|\d+','',temp[j])
        temp[j] = re.sub(r'\n|\s{2,}',' ',txt)
        sentences.append(temp[j].lower().split())
        
model = Word2Vec(sentences, iter=20, min_count=10, size=300, workers=4)

#model.save('./word2vec.txt')
#model = Word2Vec.load('./word2vec.txt')

vocab = model.wv.vocab

words = []
freq = []
for w in vocab:
    words.append(w)
    freq.append(vocab[w].count)
    
wordseries = pd.DataFrame({'word': words, 'freq': freq})
wordseries = wordseries.sort_values(['freq'], ascending = [0])
wordseries['id'] = range(1,wordseries.shape[0]+1)
wordsequence = dict(zip(wordseries['word'],wordseries['id']))

W = np.zeros((1,300))
W = np.append(W, model[wordsequence.keys()],axis=0)
W = W.astype(np.float32, copy=False)

def generate_batch(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            print(end_index)
            yield shuffled_data[start_index:end_index]
            
def blocks(data, block_size):
    data = np.array(data)
    data_size = len(data)
    nums = int((data_size-1)/block_size) + 1
    for block_num in range(nums):
        if block_num == 0:
            print("prediction start!")
        start_index = block_num * block_size
        end_index = min((block_num + 1) * block_size, data_size)
        print(end_index)
        yield data[start_index:end_index]

#Placeholders and create CNN.
filter_sizes = [2,3,4,5]
num_filters = 2
batch_size = 200
embedding_size = 300
num_filters_total = num_filters * len(filter_sizes)
sequence_length = 1403
num_epochs = 10

input_x = tf.placeholder(tf.int32, [None, sequence_length], name = "input_x")
input_y = tf.placeholder(tf.float32, [None,6], name = "input_y")
dropout_keep_prob = 0.5

embedded_chars = tf.nn.embedding_lookup(W, input_x)
embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

pooled_outputs = []
    
for i, filter_size in enumerate(filter_sizes):
        
    filter_shape = [filter_size, embedding_size, 1, num_filters]
        
    w = tf.Variable(tf.truncated_normal(filter_shape,stddev = 0.1), name = "w")
    b = tf.Variable(tf.truncated_normal([num_filters]), name = "b")
            
    conv = tf.nn.conv2d(
        embedded_chars_expanded,
        w,
        strides = [1,1,1,1],
        padding = "VALID",
        name = "conv"
    )
    h = tf.nn.relu(tf.nn.bias_add(conv, b), name = "relu")
    pooled = tf.nn.max_pool(
        h,
        ksize = [1,sequence_length - filter_size + 1, 1, 1],
        strides = [1,1,1,1],
        padding = "VALID",
        name = "pool"
    )
    pooled_outputs.append(pooled)
    
h_pool = tf.concat(pooled_outputs, 3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

wd1 = tf.Variable(tf.truncated_normal([num_filters_total, int(num_filters_total/2)], stddev=0.1), name = "wd1")
bd1 = tf.Variable(tf.truncated_normal([int(num_filters_total/2)]), name = "bd1")
layer1 = tf.nn.xw_plus_b(h_drop, wd1, bd1, name = 'layer1')
layer1 = tf.nn.relu(layer1)

wd2 = tf.Variable(tf.truncated_normal([int(num_filters_total/2),6]), name = 'wd2')
bd2 = tf.Variable(tf.truncated_normal([6]), name = "bd2")
layer2 = tf.nn.xw_plus_b(layer1, wd2, bd2, name = 'layer2')
prediction = tf.nn.softmax(layer2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = layer2, labels = input_y))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.5).minimize(loss)

df_test = pd.read_csv('test.csv')
test_input = df_test['comment_text']
for i in range(len(test_input)):
    txt = re.sub('[^\w\s\']|\d+','',test_input[i])
    txt = re.sub(r'\n|\s{2,}',' ',txt.lower())
    lst = txt.split()
    temp = []
    for word in lst:
        if word not in vocab:
            temp.append(0)
        else:
            temp.append(wordsequence[word])

    test_input[i] = temp
test_input = sequence.pad_sequences(test_input, maxlen = sequence_length)   

batches = generate_batch(list(zip(raw_input, df['A'], df['B'], df['C'], df['D'], df['E'], df['F'])), batch_size, num_epochs)
test_blocks = blocks(test_input,300)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    
    sess.run(init_op)
    
    for batch in batches:
        batch = pd.DataFrame(batch, columns = ['a','b','c','d','e','f','g'])
        x_batch = batch['a']
        y_batch = batch.loc[:, batch.columns != 'a']
        for i in range(len(x_batch)):
            txt = re.sub('[^\w\s\']|\d+','',x_batch[i])
            txt = re.sub(r'\n|\s{2,}',' ',txt.lower())
            lst = txt.split()
            temp = []
            for word in lst:
                if word not in vocab:
                    temp.append(0)
                else:
                    temp.append(wordsequence[word])
            x_batch[i] = temp
        x_batch = sequence.pad_sequences(x_batch, maxlen=sequence_length)
        #y_batch = np.array(y_batch).reshape(batch_size,6)
        _,c = sess.run([optimizer, loss],feed_dict = {input_x: x_batch, input_y: y_batch})
     
      with open('csvfile.csv', "w") as output:
          writer = csv.writer(output, lineterminator='\n')
          writer.writerow(['A','B','C','D','E','F'])
          for block in test_blocks:
              pred = sess.run(prediction, feed_dict={input_x: block})
              writer.writerows(pred)

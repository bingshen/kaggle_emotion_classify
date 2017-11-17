import tensorflow as tf
from tensorflow.contrib.data import Dataset
import pandas as pd
from make_iterator import *
from tensorflow.python.ops import lookup_ops
from Word2VecUtil import Word2VecUtil
from attention_model import AttentionModel
import numpy as np
import math,os

VAL_RATE=0.9
BATCH_SIZE=50
MAX_EPOCHS=3

def load_data(dataframe):
    train_x,train_y=[],[]
    for [review,sentiment] in dataframe[['review','sentiment']].values:
        wordlist=Word2VecUtil.review_to_wordlist(review)
        train_x.append(' '.join(wordlist))
        train_y.append(sentiment)
    return train_x,train_y

if __name__ == '__main__':
    train_df=pd.read_csv("data/labeledTrainData.tsv",delimiter="\t",quoting=3)
    src_vocab_table=lookup_ops.index_table_from_file('data/vocab.txt',default_value=0)
    train_x,train_y=load_data(train_df)
    splitVal=int(len(train_y)*VAL_RATE)
    val_x,val_y=train_x[splitVal:],train_y[splitVal:]
    src_dataset=tf.contrib.data.Dataset.from_tensor_slices((train_x,train_y))
    val_dataset=tf.contrib.data.Dataset.from_tensor_slices((val_x,val_y))
    src_iterator=train_iterator(src_dataset,src_vocab_table,BATCH_SIZE,max_length=500)
    val_iterator=train_iterator(val_dataset,src_vocab_table,BATCH_SIZE,max_length=500)
    attention_model=AttentionModel(src_iterator,val_iterator,4,200,200,False)
    sess_config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    sess_config.gpu_options.allow_growth=True
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        for i in range(MAX_EPOCHS):
            sess.run(src_iterator.initializer)
            sess.run(val_iterator.initializer)
            print("epoch:",i)
            step=0
            while True:
                try:
                    attention_model.train(sess,step)
                    step+=1
                except tf.errors.OutOfRangeError:
                    break
            step,val_preds,val_ys=0,[],[]
            while True:
                try:
                    batch_preds,batch_y=attention_model.val_infer(sess)
                    val_preds.append(batch_preds)
                    val_ys.append(batch_y)
                    correct_pred=np.equal(np.argmax(batch_preds,1),np.argmax(batch_y,1))
                    accuracy=np.mean(correct_pred)
                    print("val acc:",accuracy,"step:",step)
                    step+=1
                except tf.errors.OutOfRangeError:
                    val_preds=np.concatenate(val_preds,axis=0)
                    val_ys=np.concatenate(val_ys,axis=0)
                    correct_pred=np.equal(np.argmax(val_preds,1),np.argmax(val_ys,1))
                    accuracy=np.mean(correct_pred)
                    print("total val acc:",accuracy)
                    break;
        attention_model.save_model(sess,"model/attention_model.ckpt")
import tensorflow as tf
from tensorflow.contrib.data import Dataset
import pandas as pd
from make_iterator import *
from tensorflow.python.ops import lookup_ops
from Word2VecUtil import Word2VecUtil
from attention_model import AttentionModel
import numpy as np
import math,os

BATCH_SIZE=50

def load_data(dataframe):
    train_x,train_y=[],[]
    for [review] in dataframe[['review']].values:
        wordlist=Word2VecUtil.review_to_wordlist(review)
        train_x.append(' '.join(wordlist))
    return train_x

if __name__ == '__main__':
    test_df=pd.read_csv("data/testData.tsv",delimiter="\t",quoting=3)
    src_vocab_table=lookup_ops.index_table_from_file('data/vocab.txt',default_value=0)
    test_x=load_data(test_df)
    test_dataset=tf.contrib.data.Dataset.from_tensor_slices(test_x)
    test_iterator=infer_iterator(test_dataset,src_vocab_table,BATCH_SIZE,max_length=500)
    attention_model=AttentionModel(None,test_iterator,4,200,200,False)
    sess_config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    sess_config.gpu_options.allow_growth=True
    with tf.Session(config=sess_config) as sess:
        attention_model.load_model(sess,"model/attention_model.ckpt")
        sess.run(tf.tables_initializer())
        sess.run(test_iterator.initializer)
        test_preds=[]
        while True:
            try:
                batch_preds=attention_model.test_infer(sess)
                test_preds.append(batch_preds)
            except tf.errors.OutOfRangeError:
                test_preds=(np.concatenate(test_preds,axis=0))[:,1]
                submission=pd.DataFrame({'id':test_df['id'],'sentiment':test_preds})
                submission.to_csv('submission.csv',index=False,quoting=3)
                break
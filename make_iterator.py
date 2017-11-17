# -*- coding: UTF-8 -*-
import tensorflow as tf
import collections

# 迭代器输入数据，如果是测试数据的话，input_y填None
class BatchedInput(
    collections.namedtuple("BatchedInput",
        ("initializer","input_x","input_y","sequence_length"))):
    pass

# 目前处理的题目是二分类问题，所以独热码只会输出[1,0]或者[0,1]
def one_hot_label(y):
    trans=tf.cond(tf.equal(y,tf.constant(0)),lambda: tf.constant([1,0]),lambda: tf.constant([0,1]))
    return trans

# 组装infer的迭代器，和train的infer主要差别是没有y
def infer_iterator(infer_data,vocab_table,batch_size,max_length=None):
    eos_id=tf.cast(vocab_table.lookup(tf.constant('<eos>')),tf.int32)
    infer_data=infer_data.map(lambda x:tf.string_split([x]).values)
    if max_length:
        infer_data=infer_data.map(lambda x:x[:max_length])
    infer_data=infer_data.map(lambda x:tf.cast(vocab_table.lookup(x),tf.int32))
    infer_data=infer_data.map(lambda x:tf.reverse(x,axis=[0]))
    infer_data=infer_data.map(lambda x:(x,tf.size(x)))
    def batching_func(x):
        return x.padded_batch(batch_size,padded_shapes=(tf.TensorShape([None]),tf.TensorShape([])),padding_values=(eos_id,0))
    batched_dataset=batching_func(infer_data)
    batched_iter=batched_dataset.make_initializable_iterator()
    (infer_ids,infer_seq_len)=batched_iter.get_next()
    return BatchedInput(initializer=batched_iter.initializer,
        input_x=infer_ids,
        input_y=None,
        sequence_length=infer_seq_len)

def train_iterator(train_data,vocab_table,batch_size,max_length=None):
    eos_id=tf.cast(vocab_table.lookup(tf.constant('<eos>')),tf.int32)
    train_data=train_data.map(lambda x,y:(tf.string_split([x]).values,y))
    if max_length:
        train_data=train_data.map(lambda x,y:(x[:max_length],y))
    train_data=train_data.map(lambda x,y:(tf.cast(vocab_table.lookup(x),tf.int32),y))
    train_data=train_data.map(lambda x,y:(tf.reverse(x,axis=[0]),y))
    train_data=train_data.map(lambda x,y:(x,tf.size(x),one_hot_label(y)))
    def batching_func(x):
        return x.padded_batch(batch_size,padded_shapes=(tf.TensorShape([None]),tf.TensorShape([]),tf.TensorShape([2])),padding_values=(eos_id,0,0))
    batched_dataset=batching_func(train_data)
    batched_iter=batched_dataset.make_initializable_iterator()
    (train_ids,train_seq_len,train_labels)=batched_iter.get_next()
    return BatchedInput(initializer=batched_iter.initializer,
        input_x=train_ids,
        input_y=train_labels,
        sequence_length=train_seq_len)
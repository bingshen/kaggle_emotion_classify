import tensorflow as tf
import numpy as np
import random

def build_single_cell(num_units):
    single_cell=tf.contrib.rnn.BasicLSTMCell(num_units)
    single_cell=tf.contrib.rnn.DropoutWrapper(cell=single_cell,input_keep_prob=0.8)
    single_cell=tf.contrib.rnn.DeviceWrapper(single_cell,"/gpu:0")
    return single_cell

def build_cell_list(num_layers,num_units):
    cell_list=[]
    for i in range(num_layers):
        cell=build_single_cell(num_units)
        cell_list.append(cell)
    return tf.contrib.rnn.MultiRNNCell(cell_list)

class AttentionModel():
    def __init__(self,train_iterator,infer_iterator,num_layers,num_units,attention_size,use_attention=True):
        self.train_iterator=train_iterator
        self.infer_iterator=infer_iterator
        self.num_layers=num_layers
        self.num_units=num_units
        self.attention_size=attention_size
        self.use_attention=use_attention
        self.build_model()
    def build_model(self):
        self.input_x=tf.placeholder(shape=[None,None],dtype=tf.int32)
        self.input_y=tf.placeholder(shape=[None,2],dtype=tf.int32)
        self.seq_len=tf.placeholder(shape=[None],dtype=tf.int32)
        input_x=tf.transpose(self.input_x,[1,0])
        time_len=tf.shape(input_x)[0]
        with tf.device('/gpu:0'):
            embedding_layer=tf.Variable(tf.truncated_normal([160277,self.num_units],stddev=0.1),name="embedding_layer")
            embedding_inp=tf.nn.embedding_lookup(embedding_layer,input_x)
        half_layers=self.num_layers//2
        fw_cells=build_cell_list(half_layers,self.num_units)
        bw_cells=build_cell_list(half_layers,self.num_units)
        out_puts,_=tf.nn.bidirectional_dynamic_rnn(fw_cells,bw_cells,embedding_inp,dtype=tf.float32,sequence_length=self.seq_len,time_major=True)
        out_puts=tf.concat(out_puts,-1)
        if self.use_attention==True:
            attention_weight=tf.Variable(tf.truncated_normal([self.num_units*2,self.attention_size],stddev=0.1),name="attention_weight")
            attention_bias=tf.Variable(tf.truncated_normal([self.attention_size],stddev=0.1),name="attention_bias")
            mapping_fun1=lambda x:tf.tanh(tf.matmul(x,attention_weight)+attention_bias)
            attention_output_list=tf.map_fn(mapping_fun1,out_puts)
            alpha_weight=tf.Variable(tf.truncated_normal([self.attention_size,1],stddev=0.1),name="alpha_weight")
            mapping_fun2=lambda x:tf.matmul(x,alpha_weight)
            alpha_output_list=tf.map_fn(mapping_fun2,attention_output_list)
            alpha_logits=tf.transpose(tf.reshape(alpha_output_list,[time_len,-1]),[1,0])
            alpha=tf.nn.softmax(alpha_logits)
            alpha_trans=tf.reshape(tf.transpose(alpha,[1,0]),[time_len,-1,1])
            self.final_output=tf.reduce_sum(out_puts*alpha_trans,0)
        else:
            self.final_output=tf.reduce_mean(out_puts,0)
        self.final_weight=tf.Variable(tf.truncated_normal([2*self.num_units,2],stddev=0.1),name="final_weight")
        self.final_bais=tf.Variable(tf.truncated_normal([2],stddev=0.1),name="final_bais")
        self.logits=tf.matmul(self.final_output,self.final_weight)+self.final_bais
        self.pred=tf.nn.softmax(self.logits)
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.input_y))
        optimizer=tf.train.AdamOptimizer()
        self.loss_op=optimizer.minimize(self.loss)
        correct_pred=tf.equal(tf.argmax(self.pred,1),tf.argmax(self.input_y,1))
        self.accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        self.saver=tf.train.Saver(tf.global_variables())
    def train(self,sess,step):
        input_x,input_y,seq_len=sess.run((self.train_iterator.input_x,self.train_iterator.input_y,self.train_iterator.sequence_length))
        train_dict={self.input_x:input_x,self.input_y:input_y,self.seq_len:seq_len}
        _,loss,acc=sess.run((self.loss_op,self.loss,self.accuracy),feed_dict=train_dict)
        print("acc:",acc,"loss:",loss,"step:",step)
    def val_infer(self,sess):
        input_x,input_y,seq_len=sess.run((self.infer_iterator.input_x,self.infer_iterator.input_y,self.infer_iterator.sequence_length))
        infer_dict={self.input_x:input_x,self.seq_len:seq_len}
        return sess.run((self.pred),feed_dict=infer_dict),input_y
    def test_infer(self,sess):
        input_x,seq_len=sess.run((self.infer_iterator.input_x,self.infer_iterator.sequence_length))
        infer_dict={self.input_x:input_x,self.seq_len:seq_len}
        return sess.run((self.pred),feed_dict=infer_dict)
    def save_model(self,sess,model_name):
        self.saver.save(sess,model_name)
    def load_model(self,sess,model_name):
        self.saver.restore(sess,model_name)
    def debug(self):
        return self.loss,self.pred,self.input_y,self.logits
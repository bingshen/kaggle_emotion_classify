kaggle上的那道电影评论情感分析题，之前写过一篇解决方案。效果还很好。这里用LSTM+attention的方式在做一遍。主要是练手

题目链接：https://www.kaggle.com/c/word2vec-nlp-tutorial/data

先说效果。采用LSTM直接把各个时刻的输出求平均然后过全连接层分类。最后得分大概0.92+

采用LSTM+attention的话使用相同的网络规格和迭代次数，可以得到0.94+

也就是说，rnn对于文本分类的任务，实际上效果并不如之前实验的word2vec的方法好（之前的成绩可以达到0.97+）所以只是起一个练手的目的

网络的规格是：4层LSTM网络，每一层神经元200个。双向

使用方法：
1、创建文件夹model和文件夹data
2、下载题目数据放到data下
3、执行

	python vocab_file.py

生成词典文件
4、执行

	python run.py

训练网络并保存网络数据
5、执行

	python infer_result.py

通过保存的网络去推测测试数据，并生成答案submission.csv

------------------------------------

代码大致讲解：

lstm+attention的模型写在attention_model.py当中
可以通过参数use_attention来控制是否使用注意力模型

make_iterator当中的代码是模仿google开源的nmt系统写的
是一个用来产生训练数据和测试数据的，采用的迭代器产生，可以方便的读取比较大的数据

附：
word2vec来做这个题的解法
https://github.com/bingshen/Bag-of-Words-Meets-Bags-of-Popcorn

# coding: utf-8

# In[ ]:

import tensorflow as tf
import numpy as np

# preprocessed data
#from datasets.twitter import data
import data
import data_utils


# In[ ]:

# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='datasets/twitter/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)


# In[108]:

# parameters 
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 1024


# In[109]:

import seq2seq_wrapper


# In[110]:

import importlib
importlib.reload(seq2seq_wrapper)


# In[111]:

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/twitter/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )


# In[112]:

val_batch_gen = data_utils.rand_batch_gen(validX, validY, 256)
test_batch_gen = data_utils.rand_batch_gen(testX, testY, 256)

train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)


# In[113]:

#sess = model.train(train_batch_gen, val_batch_gen)


# In[114]:

sess = model.restore_last_session()


# In[115]:

input_ = test_batch_gen.__next__()[0]
print (input_.shape)
print (input_[0])

output = model.predict(sess, input_)
print(output.shape)


# In[116]:

replies = []
for ii, oi in zip(input_.T, output):
    q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
    decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
    if decoded.count('unk') == 0:
        if decoded not in replies:
            print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
            replies.append(decoded)


# In[117]:

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '

def handleQueryFunc(line):
	line=line.lower()
	line=data.filter_line(line, EN_WHITELIST)

	qtokenized = [line.split(' ')]
	idx_q, idx_a = data.zero_pad(qtokenized, [[]], metadata['w2idx']) 
	return idx_q


# In[ ]:

while(True):
    query = input("user :")
    idx_q = handleQueryFunc(query)
    #print (idx_q)
    output = model.predict(sess, idx_q.T)
    #print (output)
    q = data_utils.decode(sequence=idx_q[0], lookup=metadata['idx2w'], separator=' ')
    decoded = data_utils.decode(sequence=output[0], lookup=metadata['idx2w'], separator=' ').split(' ')
    
    print('***\n query : {0}; \n reply : {1}'.format(q, ' '.join(decoded)))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




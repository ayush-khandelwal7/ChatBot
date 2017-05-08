from __future__ import print_function
from __future__ import absolute_import

from flask import Flask, render_template, request, jsonify,make_response




import sys
import os
import time
from threading import Thread



#from model.chat_bot import ChatBot
#from data import parse
import tensorflow as tf
import numpy as np

# preprocessed data
#from datasets.twitter import data
import data
import data_utils
app = Flask(__name__)

#**********************************************

# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='datasets/twitter/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)


# parameters 
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 1024

import seq2seq_wrapper



import importlib
importlib.reload(seq2seq_wrapper)


model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/twitter/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )


#sess = model.train(train_batch_gen, val_batch_gen)


print ("\nRestoring Trained Model\n")
sess = model.restore_last_session()
print("Model Loaded\n")


EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '

def handleQueryFunc(line):
	line=line.lower()
	line=data.filter_line(line, EN_WHITELIST)

	qtokenized = [line.split(' ')]
	idx_q, idx_a = data.zero_pad(qtokenized, [[]], metadata['w2idx']) 
	return idx_q

#**********************************************


@app.route("/")
def hello():
    return render_template('chat.html')

@app.route("/ask", methods=['POST'])
def ask():
	message = str(request.form['messageText'])

	'''kernel = aiml.Kernel()
			
				if os.path.isfile("bot_brain.brn"):
				    kernel.bootstrap(brainFile = "bot_brain.brn")
				else:
				    kernel.bootstrap(learnFiles = os.path.abspath("aiml/std-startup.xml"), commands = "load aiml b")
				    kernel.saveBrain("bot_brain.brn")'''

	# kernel now ready for use
	while True:
	    if message == "quit":
	        exit()
	    else:
	    	idx_q = handleQueryFunc(message)
	    	#print (idx_q)
	    	output = model.predict(sess, idx_q.T)
	    	#print (output)
	    	q = data_utils.decode(sequence=idx_q[0], lookup=metadata['idx2w'], separator=' ')
	    	bot_response = data_utils.decode(sequence=output[0], lookup=metadata['idx2w'], separator=' ').split(' ')
	    	#print (bot_response)
	    	respose= ' '.join(bot_response)
	    	print (respose)
	    	return make_response(jsonify({'status':'OK','answer':respose}))

if __name__ == "__main__":  
    app.run(debug=True)
from textblob import TextBlob
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
#inp_file = 'CodeGAN-master_vs_random/codegan-pg_temp_policy_grad_full_seq_rnn_disc/input_text_neg.txt'

inp_file = 'DataAugmentation_NLP/Experiment1/policy_gradient/eval_file_pretrain_gen_59.txt'
f = open(inp_file,'r')


predicted = 0
actual = 0
for line in f.readlines():
        	text = TextBlob(line)
        	pol = text.sentiment.polarity
        	if(pol<0):
        		predicted = predicted+1
        	actual = actual + 1
acc = predicted/(actual*1.0)
f.close()

print(predicted)
print(actual)
print(acc)
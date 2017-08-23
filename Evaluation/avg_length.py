import os

import tensorflow as tf
from collections import Counter

import nltk
from nltk.util import ngrams
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

n=2

def spacer(line, postprocess=True):
        line = line.replace('\n', ' _enter ')
        line = line.replace('    ', ' _tab ')
        line = line.replace('=', ' = ')
        line = line.replace('+ =', '+=')
        line = line.replace('- =', '-=')
        line = line.replace('> =', '>=')
        line = line.replace('< =', '<=')
        line = line.replace('! =', '!=')
        line = line.replace('=  =', '==')
        line = line.replace('("', '( "')
        line = line.replace('",', '" ,')
        line = line.replace('(', ' ( ')
        line = line.replace(')', ' ) ')
        line = line.replace('[', ' [ ')
        line = line.replace(']', ' ] ')
        line = line.replace(',', ' , ')
        line = line.replace('.', ' . ')
        line = " ".join(line.split())
        if postprocess == True:
            line = line.replace('_enter','\n')
            line = line.replace('_tab', '\t')
            line = " ".join(line.split())
        line = line.strip()
        return line
reference_file_train = 'DataAugmentation_NLP/corpus_uncond_neg/input_file.txt'
#filename to evaluate
hypothesis_file = 'CodeGAN-DataAugmentation_NLP/Experiment1/policy_gradient/eval_file_reinforce_124_34.txt'
reference_file_test = 'DataAugmentation_NLP-master_vs_random/corpus_uncond_neg/target_file.txt'
#reference_file =  '../policygrad_seq-seqGAN/CodeGAN-master_test_gen_pos_also_passed/corpus3/input_file.txt'
length = 0
total = 0
with open(hypothesis_file, 'r') as f:
            #content = u' '.join(f.readlines()).decode("UTF-8").encode('ascii','ignore')
            for con in f.readlines():
                con = con.strip()
                content = con.decode("UTF-8").encode('ascii','ignore')
                hyp = nltk.word_tokenize(spacer(content))
                length = length + len(hyp)
                total = total+1

print(length)
print(total)
score = length/float(total)
print(score)

print('----------------------------------------------------------------------------------')

'''
#evaluation of actual file 
hypothesis_file = 'CodeGAN-master_vs_random/codegan-pg_temp_policy_grad_full_seq_rnn_disc/input_text_neg.txt'
#reference_file =  '../policygrad_seq-seqGAN/CodeGAN-master_test_gen_pos_also_passed/corpus3/input_file.txt'
rept = 0
total = 0
with open(hypothesis_file, 'r') as f:
            #content = u' '.join(f.readlines()).decode("UTF-8").encode('ascii','ignore')
            for con in f.readlines():
                con = con.replace('0','')
                con = con.strip()
                content = con.decode("UTF-8").encode('ascii','ignore')
                hyp = nltk.word_tokenize(spacer(content))
                length = length + len(hyp)
                total = total+1

print(length)
print(total)
score = length/float(total)
print(score)

'''
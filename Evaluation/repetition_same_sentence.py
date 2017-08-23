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
hypothesis_file = 'DataAugmentation_NLP/results/Experiment1/policy_gradient/eval_file_reinforce_120_153.txt'
reference_file_test = 'DataAugmentation_NLP/corpus_uncond_neg/target_file.txt'
#reference_file =  '../policygrad_seq-seqGAN/CodeGAN-master_test_gen_pos_also_passed/corpus3/input_file.txt'
rept = 0
total = 0
with open(hypothesis_file, 'r') as f:
            #content = u' '.join(f.readlines()).decode("UTF-8").encode('ascii','ignore')
            for con in f.readlines():
                con = con.strip()
                content = con.decode("UTF-8").encode('ascii','ignore')
                hyp = nltk.word_tokenize(spacer(content))
                trigrams_hyp = Counter(ngrams(hyp, 2))
                if(any(x>1 for x in list(trigrams_hyp.values()))):
                    rept = rept+1
                total = total+1

print(rept)
print(total)
score = rept/float(total)
print(score)

print('----------------------------------------------------------------------------------')
'''
hypothesis_file = 'DataAugmentation_NLP/codegan-pg_temp_policy_grad_full_seq_rnn_disc/input_text_neg.txt'
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
                trigrams_hyp = Counter(ngrams(hyp, 2))
                if(any(x>1 for x in list(trigrams_hyp.values()))):
                    rept = rept+1
                total = total+1

print(rept)
print(total)
score = rept/float(total)
print(score)
'''
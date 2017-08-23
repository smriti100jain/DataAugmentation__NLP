import os

import tensorflow as tf
from collections import Counter

import nltk
from nltk.util import ngrams
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

n=1

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
hypothesis_file = 'DataAugmentation_NLP/Experiment1/policy_gradient/eval_file_reinforce_124_34.txt'
reference_file_test = 'DataAugmentation_NLP/corpus_uncond_neg/target_file.txt'
#reference_file =  '../policygrad_seq-seqGAN/CodeGAN-master_test_gen_pos_also_passed/corpus3/input_file.txt'

with open(reference_file_train, 'r') as f:
            content = u' '.join(f.readlines()).decode("UTF-8").encode('ascii','ignore')
            ref = nltk.word_tokenize(spacer(content))

with open(reference_file_test, 'r') as f:
            content = u' '.join(f.readlines()).decode("UTF-8").encode('ascii','ignore')
            ref_test = nltk.word_tokenize(spacer(content))

with open(hypothesis_file, 'r') as f:
            content = u' '.join(f.readlines()).decode("UTF-8").encode('ascii','ignore')
            hyp = nltk.word_tokenize(spacer(content))
def compare(trigrams1, trigrams2):
    common=[]
    for grams1 in trigrams1:
        if grams1 in trigrams2:
            common.append(grams1)
    return common
output_file = open('test.txt','w')
for n in range(1,5):

    trigrams_hyp = set(Counter(ngrams(hyp, n)))

    trigrams_ref = set(Counter(ngrams(ref, n)))  

    trigrams_ref_test = set(Counter(ngrams(ref_test,n)))

    common_tri = len(trigrams_hyp.intersection(trigrams_ref))
    common_tri_test = len(trigrams_hyp.intersection(trigrams_ref_test))


    #common_tri = compare(trigrams_hyp.keys(), trigrams_ref.keys())
    #common_tri_test = compare(trigrams_hyp.keys(), trigrams_ref_test.keys())

    score = common_tri/float(len(trigrams_ref))
    score_test = common_tri_test/float(len(trigrams_ref_test))

    output_file.write('score:  n=  '+str(n)+' : '+str(score) + '\n')
    output_file.write('score test:  n=  '+str(n)+' : '+str(score_test) + '\n')

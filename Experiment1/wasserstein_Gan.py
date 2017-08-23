import Utils.model as model
import numpy as np
import tensorflow as tf
import random
import time
from Utils.gen_dataloader import Gen_Data_loader, Likelihood_data_loader
from Utils.dis_dataloader import Dis_dataloader
from Utils.text_classifier import TextCNN
from Utils.rollout import ROLLOUT
from Utils.target_lstm import TARGET_LSTM
import cPickle
import os
from text_generator import TextGenerator
import math
import csv
import os
import random
import numpy as np
import tensorflow as tf
import codecs
import gzip
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import unicodedata
import os
from collections import Counter
import nltk
from nltk.util import ngrams
import sys
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot
import tensorflow as tf
import gensim
import random
import pickle


pos = 0

#reload(sys)
#sys.setdefaultencoding("utf-8")




#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
EMB_DIM = 100#32
HIDDEN_DIM = 100#32
SEQ_LENGTH = 40
SEQ_LEN = SEQ_LENGTH
START_TOKEN = 0

PRE_EPOCH_NUM = 60#10#240
TRAIN_ITER = 1 # generator
SEED = 88
BATCH_SIZE = 128
##########################################################################################


#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
ITERS = 20#50#200000 # How many iterations to train for
DIM = 512#512 # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.
CRITIC_ITERS = 2#8#2 #initially tried 2 # How many critic iterations per generator iteration. We
                  # use 10 for the results in the paper, but 5 should work fine
                  # as well.
rnn_size = 300
seq_length = SEQ_LENGTH
num_layers = 2
output_keep_prob = 1.0
input_keep_prob = 1.0
LAMBDA = 100 # Gradient penalty lambda hyperparameter.
                         


# Training parameters
dis_batch_size = 64
dis_num_epochs = 2
dis_alter_epoch = 60#10#25#50



if(pos == 1):
    positive_file = 'save_pos_was/real_data.txt'
    negative_file = 'target_generate_pos_was/generator_sample.txt'
    eval_file = 'target_generate_pos_was/eval_file.txt'
    test_file = 'target_generate_pos_was/test_file.txt'
    save_dir = 'save_models_pos'
    vocabulary_file = '../corpus_uncond_pos/index2word.pickle'
    index2word = cPickle.load(open(vocabulary_file, "rb"))
    word2index = cPickle.load(open('../corpus_uncond_pos/word2index.pickle'))
    inp_ref_file = 'input_text_pos.txt'
    test_ref_file = 'target_text_pos.txt'
    starting_word_file = 'target_generate_pos_was/start_word_file.txt'
else:
    positive_file = 'save_neg_was/real_data.txt'
    negative_file = 'target_generate_neg_was/generator_sample.txt'
    eval_file = 'target_generate_neg_was/eval_file.txt'
    test_file = 'target_generate_neg_was/test_file.txt'
    save_dir = 'save_models_neg'
    vocabulary_file = '../corpus_uncond_neg/index2word.pickle'
    index2word = cPickle.load(open(vocabulary_file, "rb"))
    word2index = cPickle.load(open('../corpus_uncond_neg/word2index.pickle'))
    inp_ref_file = 'input_text_neg.txt'
    test_ref_file = 'target_text_neg.txt'
    starting_word_file = 'target_generate_neg_was/start_word_file.txt'


wds = ' '.join(word2index.keys())
text = nltk.word_tokenize(wds)
tags = nltk.pos_tag(text)
remove_words_ind = []
remove_words = []
for word,tag in tags:
    if(tag=='CD' or tag=='JJ' or tag=='.'):
        remove_words_ind.append(word2index[word])
        remove_words.append(word)

good_ids = list(set(index2word.keys())-set(remove_words_ind)-set([0]))


vocab_size = len(word2index.keys())

##############################################################################################


class PoemGen(model.LSTM):
    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer()  # ignore learning rate


def get_trainable_model(num_emb):
    return PoemGen(num_emb, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN,good_ids,pos)


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file,data_loader):
    #  Generated Samples
    generated_samples = []
    start = time.time()
    data_loader.reset_pointer()
    for _ in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        generated_samples.extend(trainable_model.generate(sess,batch))
    end = time.time()
    # print 'Sample generation time:', (end - start)

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            #buffer = u''.join([index2word[x] for x in poem]).encode('utf-8') + '\n'
            fout.write(buffer)

def generate_samples2(sess, trainable_model, batch_size, generated_num, output_file,data_loader):
    #  Generated Samples
    generated_samples = []
    start = time.time()
    data_loader.reset_pointer()
    for _ in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        generated_samples.extend(trainable_model.generate(sess,batch))
    end = time.time()
    # print 'Sample generation time:', (end - start)

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            #buffer = ' '.join([str(x) for x in poem]) + '\n'
            buffer = u' '.join([index2word[x] for x in poem]).encode('utf-8') + '\n'
            fout.write(buffer)
    fout.close()
    
    
def target_loss(sess, target_lstm, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)




def pre_train_epoch(sess, trainable_model, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss, g_pred = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)
def compare(trigrams1, trigrams2):
    common=[]
    for grams1 in trigrams1:
        if grams1 in trigrams2:
            common.append(grams1)
    return common


def ResBlock(name, inputs):
    output = inputs
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.1', DIM, DIM, 5, output)
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.2', DIM, DIM, 5, output)
    return inputs + (0.3*output)

def Discriminator(inputs):
    output = tf.transpose(inputs, [0,2,1])
    output = lib.ops.conv1d.Conv1D('Discriminator.Input', vocab_size, DIM, 1, output)
    output = ResBlock('Discriminator.1', output)
    output = ResBlock('Discriminator.2', output)
    output = ResBlock('Discriminator.3', output)
    output = ResBlock('Discriminator.4', output)
    output = ResBlock('Discriminator.5', output)
    output = tf.reshape(output, [-1, SEQ_LEN*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', SEQ_LEN*DIM, 1, output)
    return output



def main():
    random.seed(SEED)
    np.random.seed(SEED)
    if(pos==1):
        stringGenerator = TextGenerator('../corpus_uncond_pos/index2word.pickle', '../corpus_uncond_pos/word2index.pickle', '../corpus_uncond_pos/input_file.txt','../corpus_uncond_pos/target_file.txt','../corpus_uncond_pos/vocab_creation_file.txt')
    else:
        stringGenerator = TextGenerator('../corpus_uncond_neg/index2word.pickle', '../corpus_uncond_neg/word2index.pickle', '../corpus_uncond_neg/input_file.txt','../corpus_uncond_neg/target_file.txt','../corpus_uncond_neg/vocab_creation_file.txt')
    generated_num_inp = stringGenerator.sentencesCount_inp
    generated_num_test = stringGenerator.sentencesCount_test
    
    with open(starting_word_file, "w+") as op:
        for i in range(len(good_ids)):
            tokensSequence = [good_ids[i]]
            tokensSequence += [0] * (SEQ_LENGTH-1)
            strSentence = " ".join([str(index) for index in tokensSequence]) + "\n"
            op.write(strSentence)
      


    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE,SEQ_LENGTH)
    start_data_loader = Gen_Data_loader(BATCH_SIZE,SEQ_LENGTH)
    likelihood_data_loader = Likelihood_data_loader(BATCH_SIZE,SEQ_LENGTH)
    vocab_size = len(stringGenerator.index2Word)
    dis_data_loader = Dis_dataloader(SEQ_LENGTH)
    #Embedding matrix from google vec:
    GLOVE_DIR = '../corpus_uncond_neg/glove.6B/'
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    EmbeddingMatrix = np.zeros((vocab_size,EMB_DIM))



    #embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for  i,word in index2word.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            EmbeddingMatrix[i] = embedding_vector
        else:
            EmbeddingMatrix[i] = np.random.uniform(-1,1,EMB_DIM)
    if(pos==1):
        np.savez('embedding_pos.npz',EmbeddingMatrix)
    else:
        np.savez('embedding_neg.npz',EmbeddingMatrix)
    ###############################################################################

    best_score = 1000
    generator = get_trainable_model(vocab_size)


    real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, SEQ_LEN])
    real_inputs = tf.one_hot(real_inputs_discrete, vocab_size)
    print(real_inputs)

    disc_real = Discriminator(real_inputs) 
    disc_fake = Discriminator(generator.g_predictions_wgan)

    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    gen_cost = -tf.reduce_mean(disc_fake)

    # WGAN lipschitz-penalty
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1,1], 
        minval=0.,
        maxval=1.
    )
    differences = generator.g_predictions_wgan - real_inputs
    interpolates = real_inputs + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty
    

    gen_params = generator.g_params

    disc_params = lib.params_with_name('Discriminator')


    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)


    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    #generate_samples(sess, target_lstm, 64, 10000, positive_file)
    stringGenerator.saveSamplesToFile_inp(SEQ_LENGTH, generated_num_inp, positive_file)
    stringGenerator.saveSamplesToFile_inp_text(SEQ_LENGTH, generated_num_inp, inp_ref_file)
    stringGenerator.saveSamplesToFile_test_text(SEQ_LENGTH, generated_num_test, test_ref_file)

    stringGenerator.saveSamplesToFile_test(SEQ_LENGTH, generated_num_test, test_file)
    gen_data_loader.create_batches(positive_file)

    start_data_loader.create_batches(starting_word_file)

   

    if(pos == 1):
        log = open('log_pos_was/experiment-log.txt', 'w')
    else:
        log = open('log_neg_was/experiment-log.txt','w')
    #  pre-train generator
    print 'Start pre-training...'
    log.write('pre-training...\n')


    EPOCHS = 0
    load = 0
    if(load == 1):
        epoch = 5
        if(pos == 1):
            saver.restore(sess, "/target_generate_pos_was/pretrain"+str(epoch)+".ckpt")
        else:
            saver.restore(sess, "/target_generate_pos_was/pretrain"+str(epoch)+".ckpt")
        EPOCHS = EPOCHS + epoch

    for epoch in xrange(PRE_EPOCH_NUM):
        print 'pre-train epoch:', epoch
        if(pos==1):
            eval_file2 = 'target_generate_pos_was/eval_file'+'_pretrain_gen_'+str(EPOCHS+epoch)+'.txt'
        else:
            eval_file2 = 'target_generate_neg_was/eval_file'+'_pretrain_gen_'+str(EPOCHS+epoch)+'.txt'

        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            generate_samples2(sess, generator, BATCH_SIZE, len(good_ids), eval_file2,start_data_loader)
            generate_samples(sess, generator, BATCH_SIZE, len(good_ids), eval_file,start_data_loader)
      
        
            likelihood_data_loader.create_batches(positive_file)
            train_loss = target_loss(sess, generator, likelihood_data_loader)
            likelihood_data_loader.create_batches(test_file)
            test_loss = target_loss(sess, generator, likelihood_data_loader)
            print 'pre-train epoch ', epoch, 'test_loss ', test_loss,'train_loss',train_loss
            buffer = str(epoch) + ' test_loss : ' + str(test_loss) + ' train_loss : ' + str(train_loss)+'\n'
            log.write(buffer)
            if(pos==1):
                saver.save(sess,'target_generate_pos_was/pretrain',global_step=EPOCHS+epoch)
            else:
                saver.save(sess,'target_generate_neg_was/pretrain',global_step=EPOCHS+epoch)
        
    if(pos ==1):
        eval_file2 = 'target_generate_pos_was/eval_file'+'_pretrain_gen_'+str(EPOCHS+epoch)+'.txt'
    else:
        eval_file2 = 'target_generate_neg_was/eval_file'+'_pretrain_gen_'+str(EPOCHS+epoch)+'.txt'

    generate_samples2(sess, generator, BATCH_SIZE, len(good_ids), eval_file2,start_data_loader)
    likelihood_data_loader.create_batches(positive_file)
    train_loss = target_loss(sess, generator, likelihood_data_loader)
    likelihood_data_loader.create_batches(test_file)
    test_loss = target_loss(sess, generator, likelihood_data_loader)
    print 'pre-train epoch ', epoch, 'test_loss ', test_loss,'train_loss',train_loss
    buffer = str(epoch) + ' test_loss : ' + str(test_loss) + ' train_loss : ' + str(train_loss)+'\n'
    log.write(buffer)


    def batch_iter(data, batch_size, num_epochs):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield np.array(shuffled_data[start_index:end_index],dtype='int32')

    def load_train_data( file):
        """
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        examples = []
        with open(file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                examples.append(parse_line)
        return np.array(examples)


    EPOCHS  = EPOCHS + PRE_EPOCH_NUM
    print 'Start training discriminator...'
    for epoch in range(dis_alter_epoch):
        print('disctrainingepoch: '+str(epoch))

        #  train discriminator
        pos_data = load_train_data(positive_file)
        pos_batches = batch_iter(pos_data, BATCH_SIZE, 1)
        for i in range(int(len(pos_data) / BATCH_SIZE) + 1):
            A=pos_batches.next()
            if(np.shape(A)[0]==BATCH_SIZE):
                _disc_cost, _ = sess.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_inputs_discrete:A}
                )
            else:
                break
        if(epoch%30==0):
            if(pos==1):
                saver.save(sess,'target_generate_pos_was/disc',global_step=EPOCHS+epoch)
            else:
                saver.save(sess,'target_generate_neg_was/disc',global_step=EPOCHS+epoch)
        
            
    

    EPOCHS = EPOCHS + dis_alter_epoch

    for iteration in xrange(ITERS):
        start_time = time.time()
        print 'training wgan...'
        #  train discriminator
        pos_data = load_train_data(positive_file)
        pos_batches = batch_iter(pos_data, BATCH_SIZE, 1)
        
        # Train generator
        
        for ii in range(int(len(pos_data) / BATCH_SIZE) + 1):
            A = pos_batches.next()
            if(np.shape(A)[0]==BATCH_SIZE):
                if iteration > 0:
                    _gen_cost,_ = sess.run([gen_cost,gen_train_op],feed_dict={real_inputs_discrete:A})
                # Train critic
                for pp in xrange(CRITIC_ITERS):
                    _disc_cost, _ = sess.run([disc_cost, disc_train_op],
                        feed_dict={real_inputs_discrete:A}
                    )
            else:
                break

            if ii % 10 == 0:
                if(pos == 1):
                    eval_file2 = 'target_generate_pos_was/eval_file_reinforce_'+str(EPOCHS+iteration)+'_'+str(ii)+'.txt'
                else:
                    eval_file2 = 'target_generate_neg_was/eval_file_reinforce_'+str(EPOCHS+iteration)+'_'+str(ii)+'.txt'
                
                generate_samples2(sess, generator, BATCH_SIZE, len(good_ids), eval_file2,start_data_loader)
                generate_samples(sess, generator, BATCH_SIZE, len(good_ids), eval_file,start_data_loader)
                hyp = []
                
                likelihood_data_loader.create_batches(positive_file)
                train_loss = target_loss(sess, generator, likelihood_data_loader)
                likelihood_data_loader.create_batches(test_file)
                test_loss = target_loss(sess, generator, likelihood_data_loader)
                print 'reinf-train epoch ', iteration, 'test_loss ', test_loss,'train_loss',train_loss,'disc_cost',_disc_cost
                buffer = str(iteration) + ' test_loss : ' + str(test_loss) + ' train_loss : ' + str(train_loss)+ ' _disc_cost '+ str(_disc_cost)+'\n'

                log.write(buffer)

   
    log.close()


if __name__ == '__main__':
    main()



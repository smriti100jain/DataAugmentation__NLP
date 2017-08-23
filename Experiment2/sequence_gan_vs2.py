import Utils.model as model
import numpy as np
import tensorflow as tf
import random
import time
from Utils.gen_dataloader import Gen_Data_loader, Likelihood_data_loader
from Utils.dis_dataloader import Dis_dataloader
from Utils.text_classifier import TextLSTM
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



pos = 0

#reload(sys)
#sys.setdefaultencoding("utf-8")




#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
EMB_DIM = 100#32
HIDDEN_DIM = 100#32
SEQ_LENGTH = 40
START_TOKEN = 0

PRE_EPOCH_NUM = 60#10#240
TRAIN_ITER = 1 # generator
SEED = 88
BATCH_SIZE = 128
##########################################################################################

TOTAL_BATCH = 5#10#500

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 100#64
dis_num_hidden = 100
dis_num_layers = 2
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2

# Training parameters
dis_batch_size = 64
dis_num_epochs = 2
dis_alter_epoch = 100#10#25#50
if(pos == 1):
    positive_file = 'save_pos/real_data.txt'
    negative_file = 'target_generate_pos/generator_sample.txt'
    eval_file = 'target_generate_pos/eval_file.txt'
    test_file = 'target_generate_pos/test_file.txt'
    save_dir = 'save_models_pos'
    vocabulary_file = '../corpus_uncond_pos/index2word.pickle'
    index2word = cPickle.load(open(vocabulary_file, "rb"))
    word2index = cPickle.load(open('../corpus_uncond_pos/word2index.pickle'))
    inp_ref_file = 'input_text_pos.txt'
    test_ref_file = 'target_text_pos.txt'
    starting_word_file = 'target_generate_pos/start_word_file.txt'
else:
    positive_file = 'save_neg/real_data.txt'
    negative_file = 'target_generate_neg/generator_sample.txt'
    eval_file = 'target_generate_neg/eval_file.txt'
    test_file = 'target_generate_neg/test_file.txt'
    save_dir = 'save_models_neg'
    vocabulary_file = '../corpus_uncond_neg/index2word.pickle'
    index2word = cPickle.load(open(vocabulary_file, "rb"))
    word2index = cPickle.load(open('../corpus_uncond_neg/word2index.pickle'))
    inp_ref_file = 'input_text_neg.txt'
    test_ref_file = 'target_text_neg.txt'
    starting_word_file = 'target_generate_neg/start_word_file.txt'


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


    with tf.variable_scope('discriminator'):
        cnn = TextLSTM(
            sequence_length=SEQ_LENGTH,
            num_classes=2,
            vocab_size=vocab_size,
            embedding_size=dis_embedding_dim,
            num_hidden=dis_num_hidden,
            num_layers=dis_num_layers,pos=pos,
            BATCH_SIZE=BATCH_SIZE,start_token=START_TOKEN,l2_reg_lambda=dis_l2_reg_lambda)

    cnn_params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
    # Define Discriminator Training procedure
    dis_global_step = tf.Variable(0, name="global_step", trainable=False)
    dis_optimizer = tf.train.AdamOptimizer(1e-4)
    dis_grads_and_vars = dis_optimizer.compute_gradients(cnn.loss, cnn_params, aggregation_method=2)
    dis_train_op = dis_optimizer.apply_gradients(dis_grads_and_vars, global_step=dis_global_step)

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

    print('Hey........................................')
    start_data_loader.create_batches(starting_word_file)
    print('ehyssl')

    


    if(pos == 1):
        log = open('log_pos/experiment-log.txt', 'w')
    else:
        log = open('log_neg/experiment-log.txt','w')
    #  pre-train generator
    print 'Start pre-training...'
    log.write('pre-training...\n')


    EPOCHS = 0
    load = 0
    if(load == 1):
        epoch = 5
        if(pos == 1):
            path = tf.train.get_checkpoint_state('target_generate_pos')
            print(path)
            saver.restore(sess, path.model_checkpoint_path)
            #saver = tf.train.import_meta_graph("target_generate_pos/disc-90.meta")
            #saver.restore(sess,"target_generate_pos/disc-90.ckpt")
        else:
            path = tf.train.get_checkpoint_state('target_generate_neg')
            print(path)
            saver.restore(sess, path.model_checkpoint_path)
        EPOCHS = EPOCHS + PRE_EPOCH_NUM+ dis_alter_epoch
        
    for epoch in xrange(0,PRE_EPOCH_NUM):
        print 'pre-train epoch:', epoch
        if(pos==1):
            eval_file2 = 'target_generate_pos/eval_file'+'_pretrain_gen_'+str(EPOCHS+epoch)+'.txt'
        else:
            eval_file2 = 'target_generate_neg/eval_file'+'_pretrain_gen_'+str(EPOCHS+epoch)+'.txt'

        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            generate_samples2(sess, generator, BATCH_SIZE, len(good_ids), eval_file2,start_data_loader)
            generate_samples(sess, generator, BATCH_SIZE, len(good_ids), eval_file,start_data_loader)
            per_tri_test = 0
            per_di_test = 0
            per_quad_test = 0
            per_di = 0
            per_tri = 0
            per_quad = 0
           

            likelihood_data_loader.create_batches(positive_file)
            train_loss = target_loss(sess, generator, likelihood_data_loader)
            likelihood_data_loader.create_batches(test_file)
            test_loss = target_loss(sess, generator, likelihood_data_loader)
            print 'pre-train epoch ', epoch, 'test_loss ', test_loss,'train_loss',train_loss,'per_di',per_di,'per_quad',per_quad,'per_tri',per_tri,'per_di_test',per_di_test,'per_quad_test',per_quad_test,'per_tri_test',per_tri_test
            buffer = str(epoch) + ' test_loss : ' + str(test_loss) + ' train_loss : ' + str(train_loss)+ 'per_di : '+str(per_di)+'per_quad : '+str(per_quad)+'per_tri'+str(per_tri)+'per_di : '+str(per_di_test)+'per_quad : '+str(per_quad_test)+'per_tri'+str(per_tri_test)+'\n'
            log.write(buffer)
            if(pos==1):
                saver.save(sess,'target_generate_pos/pretrain',global_step=EPOCHS+epoch)
            else:
                saver.save(sess,'target_generate_neg/pretrain',global_step=EPOCHS+epoch)
        
    if(pos ==1):
        eval_file2 = 'target_generate_pos/eval_file'+'_pretrain_gen_'+str(EPOCHS+epoch)+'.txt'
    else:
        eval_file2 = 'target_generate_neg/eval_file'+'_pretrain_gen_'+str(EPOCHS+epoch)+'.txt'

    generate_samples2(sess, generator, BATCH_SIZE, len(good_ids), eval_file2,start_data_loader)
    likelihood_data_loader.create_batches(positive_file)
    train_loss = target_loss(sess, generator, likelihood_data_loader)
    likelihood_data_loader.create_batches(test_file)
    test_loss = target_loss(sess, generator, likelihood_data_loader)
    print 'pre-train epoch ', epoch, 'test_loss ', test_loss,'train_loss',train_loss
    buffer = str(epoch) + ' testloss : ' + str(test_loss) + ' trainloss : ' + str(train_loss)+'\n'
    log.write(buffer)


    EPOCHS  = EPOCHS + PRE_EPOCH_NUM


    print 'Start training discriminator...'
    for epoch in range(0):#dis_alter_epoch):
        print('disctrainingepoch: '+str(epoch))
        #generate from start same as actual data
        generate_samples(sess, generator, BATCH_SIZE, len(good_ids), negative_file,start_data_loader)

        #  train discriminator
        dis_x_train, dis_y_train = dis_data_loader.load_train_data(positive_file, negative_file)
        dis_batches = dis_data_loader.batch_iter(
            zip(dis_x_train, dis_y_train), dis_batch_size, dis_num_epochs
        )

        for batch in dis_batches:
            try:
                x_batch, y_batch = zip(*batch)
                feed = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: dis_dropout_keep_prob
                }
                _, step = sess.run([dis_train_op, dis_global_step], feed)
            except ValueError:
                pass
        if(epoch%30==0):
            if(pos==1):
                saver.save(sess,'target_generate_pos/disc',global_step=EPOCHS+epoch)
            else:
                saver.save(sess,'target_generate_neg/disc',global_step=EPOCHS+epoch)
        
        



    EPOCHS  = EPOCHS + dis_alter_epoch
    
    rollout = ROLLOUT(generator, 0.8)

    print '#########################################################################'
    print 'Start Reinforcement Training Generator...'
    log.write('Reinforcement Training...\n')
    per_tri_test = 0
    per_di_test = 0
    per_quad_test = 0
    per_di = 0
    per_tri = 0
    per_quad = 0
            
    for total_batch in range(TOTAL_BATCH):
        gen_data_loader.reset_pointer()
        
        for it in xrange(start_data_loader.num_batch):
            print('start gen training')
            batch = gen_data_loader.next_batch()
            samples = generator.generate(sess,batch)
            rewards = rollout.get_reward(sess, samples, 16, cnn)
            feed = {generator.x: samples, generator.rewards: rewards}
            _, g_loss = sess.run([generator.g_updates, generator.g_loss], feed_dict=feed)
            if(pos == 1):
                    eval_file2 = 'target_generate_pos/eval_file_reinforce_'+str(EPOCHS+total_batch)+'_'+str(it)+'.txt'
            else:
                    eval_file2 = 'target_generate_neg/eval_file_reinforce_'+str(EPOCHS+total_batch)+'_'+str(it)+'.txt'

            generate_samples2(sess, generator, BATCH_SIZE, len(good_ids), eval_file2,start_data_loader)
            generate_samples(sess, generator, BATCH_SIZE, len(good_ids), eval_file,start_data_loader)
                
            
            likelihood_data_loader.create_batches(positive_file)
            train_loss = target_loss(sess, generator, likelihood_data_loader)
            likelihood_data_loader.create_batches(test_file)
            test_loss = target_loss(sess, generator, likelihood_data_loader)
            print 'reinf-train epoch ', total_batch, 'test_loss ', test_loss,'train_loss',train_loss,'per_di',per_di,'per_quad',per_quad,'per_tri',per_tri,'per_di_test',per_di_test,'per_quad_test',per_quad_test,'per_tri_test',per_tri_test
            buffer = str(total_batch) + ' test_loss : ' + str(test_loss) + ' train_loss : ' + str(train_loss)+ 'per_di : '+str(per_di)+'per_quad : '+str(per_quad)+'per_tri'+str(per_tri)+'per_di : '+str(per_di_test)+'per_quad : '+str(per_quad_test)+'per_tri'+str(per_tri_test)+'\n'
                
            log.write(buffer)
            
            rollout.update_params()
            #here i generate samples with start_data_loader
            generate_samples(sess, generator, BATCH_SIZE, len(good_ids), negative_file,start_data_loader)
            dis_x_train, dis_y_train = dis_data_loader.load_train_data(positive_file, negative_file)
            dis_batches = dis_data_loader.batch_iter(zip(dis_x_train, dis_y_train), dis_batch_size, 3)
            for batch2 in dis_batches:
                try:
                    x_batch, y_batch = zip(*batch2)
                    feed = {cnn.input_x: x_batch,cnn.input_y: y_batch,cnn.dropout_keep_prob: dis_dropout_keep_prob}
                    _, step = sess.run([dis_train_op, dis_global_step], feed)
                except ValueError:
                    pass
        
            
    log.close()

if __name__ == '__main__':
    main()



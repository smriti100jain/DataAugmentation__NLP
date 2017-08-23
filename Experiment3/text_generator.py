import re
import nltk
import cPickle as pickle
from random import randint
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import re



class TextGenerator:
    def __init__(self, index2WordFile, word2IndexFile, corpus_inp,corpus_test,corpus_tot, sentenceLengthLimit = 20):
        self.index2WordFile = index2WordFile
        self.word2IndexFile = word2IndexFile
        self.sequenceOfIndices = []
        self.sequenceOfIndices_inp = []
        self.sequenceOfIndices_test = []
        self.corpusWordsCount_inp = None
        self.corpusWordsCount_test = None
        self.word2Index = {}
        self.index2Word = {}
        self.corpusWordsCount = None

        with open(corpus_tot, 'r') as f:
            content = u' '.join(f.readlines()).decode("UTF-8").encode('ascii','ignore')
            self.sentences = []
            self.sentencesCount = 0
            # TODO process blocks instead of lines
            self.sentences_inp, self.sentencesCount_inp = self.splitBlocks(corpus_inp)
            self.sentences_test, self.sentencesCount_test = self.splitBlocks(corpus_test)
            print(self.sentences_inp[1])
            words = nltk.word_tokenize(self.spacer(content))
            self.tokens = [word.lower() for word in words]
            self.word2Index[''] = [0]
            self.index2Word[0] = ''
            for word in self.tokens:
                if word in self.word2Index:
                    index = self.word2Index[word]
                else:
                    index = len(self.word2Index)
                    self.word2Index[word] = index
                    self.index2Word[index] = word
                self.sequenceOfIndices.append(index)

            with open(index2WordFile, 'wb') as handle:
                pickle.dump(self.index2Word, handle)

            with open(word2IndexFile, 'wb') as handle:
                pickle.dump(self.word2Index, handle)
        '''
        self.corpusWordsCount = len(self.tokens)
        with open(corpus_inp, 'r') as f:
            content = ' '.join(f.readlines())
            words = nltk.word_tokenize(self.spacer(content))
            self.tokens_inp = [word.lower() for word in words]
            self.corpusWordsCount_inp = len(self.tokens_inp)
            for word in self.tokens_inp:
                index = self.word2Index[word]
                self.sequenceOfIndices_inp.append(index)
        with open(corpus_test, 'r') as f:
            content = ' '.join(f.readlines())
            words = nltk.word_tokenize(self.spacer(content))
            self.tokens_test = [word.lower() for word in words]
            self.corpusWordsCount_test = len(self.tokens_test)
            for word in self.tokens_test:
                index = self.word2Index[word]
                self.sequenceOfIndices_test.append(index)
        '''

    def spacer(self, line, postprocess=True):
        line = re.sub('[^a-zA-Z0-9\n\.\!\'\,]', ' ', line)
        line = line.replace('...', ' . ')
        line = line.replace('..', ' . ')
        line = line.replace('....', ' .')
        line = line.replace('!!', ' ! ')
        line = line.replace('!!!', ' ! ')
        line = line.replace('!!!!', ' ! ')
        line = line.replace('=  =', '==')
        line = line.replace('("', '( "')
        line = line.replace('",', '" ,')
        line = line.replace('(', ' ( ')
        line = line.replace(')', ' ) ')
        line = line.replace('[', ' [ ')
        line = line.replace(']', ' ] ')
        line = line.replace(',', ' , ')
        line = line.replace('.', ' . ')
        line = line.replace('br ','')
        line = " ".join(line.split())
        line = line.strip()
        return line
    '''
    def  splitBlocks(self, f):
        blocks = []
        block = []
        count = 0
        for i, line in enumerate(f):
            block += nltk.word_tokenize(self.spacer(line))
            if len(line) == 1 and line == '\n':
                blocks.append(block)
                count += 1
                block = []
        return blocks, count

    def generateSequence(self, length):
        if self.sentencesCount == 0:
            maxIndex = self.corpusWordsCount - length
            startIndex = randint(0, maxIndex)
            return self.sequenceOfIndices[startIndex: startIndex + length]
        else:
            sentenceIndex = randint(0, self.sentencesCount - 1)
            wordsSequence = self.sentences[sentenceIndex]
            tokensSequence = [self.word2Index[word.lower()] for word in wordsSequence if word in self.word2Index]
            spacesToAppend = length - len(tokensSequence)
            if spacesToAppend > 0:
                tokensSequence += [0] * spacesToAppend
            return tokensSequence

    def saveSamplesToFile(self, length, samplesCount, fileToSave):
        samples = [self.generateSequence(length) for _ in range(samplesCount)]
        with open(fileToSave, "w+") as text_file:
            for sample in samples:
                strSentence = " ".join([str(index) for index in sample]) + "\n"
                text_file.write(strSentence)
    '''

    def splitBlocks(self, corpus):
        blocks = []
        block = []
        count = 0
        with open(corpus, mode='r') as f:
            #file = f.readlines()
            content = u' '.join(f.readlines()).decode("UTF-8").encode('ascii','ignore')
            tokenized = nltk.tokenize.sent_tokenize(self.spacer(content))
        temp = []
        for i in tokenized:
            words = nltk.word_tokenize(i)
            if(len(words)>2):
                temp.append(i)
            else:
                print(words)
        return temp, len(temp)
   
    def generateSequence_inp(self, length,sentenceIndex):
        wordsSequence = nltk.word_tokenize(self.sentences_inp[sentenceIndex])
        tokensSequence = [self.word2Index[word.lower()] for word in wordsSequence if word in self.word2Index]
        spacesToAppend = length - len(tokensSequence)
        if spacesToAppend > 0:
                tokensSequence += [0] * spacesToAppend
        else:
            tokensSequence = tokensSequence[:length]
        return tokensSequence

    def generateSequence_inp_txt(self, length,sentenceIndex):
        wordsSequence = nltk.word_tokenize(self.sentences_inp[sentenceIndex])
        tokensSequence = [word.lower() for word in wordsSequence if word in self.word2Index]
        spacesToAppend = length - len(tokensSequence)
        if spacesToAppend > 0:
                tokensSequence += [0] * spacesToAppend
        else:
            tokensSequence = tokensSequence[:length]
        return tokensSequence

    def generateSequence_test_txt(self, length,sentenceIndex):
        wordsSequence = nltk.word_tokenize(self.sentences_test[sentenceIndex])
        tokensSequence = [word.lower() for word in wordsSequence if word in self.word2Index]
        spacesToAppend = length - len(tokensSequence)
        if spacesToAppend > 0:
                tokensSequence += [0] * spacesToAppend
        else:
            tokensSequence = tokensSequence[:length]
        return tokensSequence

    def saveSamplesToFile_inp(self, length, samplesCount, fileToSave):
        samples = [self.generateSequence_inp(length,i) for i in range(self.sentencesCount_inp)]
        with open(fileToSave, "w+") as text_file:
            for sample in samples:
                strSentence = " ".join([str(index) for index in sample]) + "\n"
                text_file.write(strSentence)

    def saveSamplesToFile_inp_text(self, length, samplesCount, fileToSave):
        samples = [self.generateSequence_inp_txt(length,i) for i in range(self.sentencesCount_inp)]
        with open(fileToSave, "w+") as text_file:
            for sample in samples:
                strSentence = " ".join([str(index) for index in sample]) + "\n"
                text_file.write(strSentence)

    def saveSamplesToFile_test_text(self, length, samplesCount, fileToSave):
        samples = [self.generateSequence_test_txt(length,i) for i in range(self.sentencesCount_test)]
        with open(fileToSave, "w+") as text_file:
            for sample in samples:
                strSentence = " ".join([str(index) for index in sample]) + "\n"
                text_file.write(strSentence)

    def generateSequence_test(self, length,sentenceIndex):
        wordsSequence = nltk.word_tokenize(self.sentences_test[sentenceIndex])
        tokensSequence = [self.word2Index[word.lower()] for word in wordsSequence if word in self.word2Index]
        spacesToAppend = length - len(tokensSequence)
        if spacesToAppend > 0:
                tokensSequence += [0] * spacesToAppend
        else:
            tokensSequence = tokensSequence[:length]
        return tokensSequence

    def saveSamplesToFile_test(self, length, samplesCount, fileToSave):
        samples = [self.generateSequence_test(length,i) for i in range(self.sentencesCount_test)]
        with open(fileToSave, "w+") as text_file:
            for sample in samples:
                strSentence = " ".join([str(index) for index in sample]) + "\n"
                text_file.write(strSentence)


    def getTextFromTokenSequence(self, lineOfTokens):
        strWords = []
        indices = [int(strIndex) for strIndex in lineOfTokens.split(" ")]
        words = [self.index2Word[index] for index in indices if index in self.index2Word]
        for word in words:
            if word.strip() == "_enter":
                strWords.append('\n')
            elif word.strip() == "_tab":
                strWords.append('\t')
            else:
                strWords.append(word)
        return " ".join(strWords)


        #strWords = " ".join(words)
if __name__ == "__main__":
    generator = TextGenerator('../corpus_uncond_neg/index2word.pickle', '../corpus_uncond_neg/word2index.pickle', '../corpus_uncond_neg/input_file.txt','../corpus_uncond_neg/target_file.txt','../corpus_uncond_neg/vocab_creation_file.txt')
    startind = 0
    for i in range(20):
        testSequenceIndices = generator.generateSequence_inp(50,startind)
        startind = startind + 20
        testSequenceWords = [generator.index2Word[index] for index in testSequenceIndices if index in generator.index2Word]
        i = i+20
        print(" ".join(testSequenceWords))


   


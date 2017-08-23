import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")


pos = 0
if(pos == 1):
	foldersave = '../corpus_uncond_pos'
else:
	foldersave = '../corpus_uncond_neg'

if(pos==1):
	GAN_DATA_test = '../../../../../../../opt/ADL_db/Users/sjain/NLP/GAN_DATA/test/pos'
	GAN_DATA_train = '../../../../../../../opt/ADL_db/Users/sjain/NLP/GAN_DATA/train/pos'
else:
	GAN_DATA_test = '../../../../../../../opt/ADL_db/Users/sjain/NLP/GAN_DATA/test/neg'
	GAN_DATA_train = '../../../../../../../opt/ADL_db/Users/sjain/NLP/GAN_DATA/train/neg'

filenames = os.listdir(os.path.join(GAN_DATA_train))
with open(os.path.join(foldersave,'input_file.txt'), 'w') as outfile:
    for fname in filenames:
        with open(os.path.join(GAN_DATA_train,fname)) as infile:
            for line in infile:
                outfile.write(line)

filenames = os.listdir(os.path.join(GAN_DATA_test))
with open(os.path.join(foldersave,'target_file.txt'), 'w') as outfile:
    for fname in filenames:
        with open(os.path.join(GAN_DATA_test,fname)) as infile:
            for line in infile:
                outfile.write(line)

with open(os.path.join(foldersave,'vocab_creation_file.txt'), 'w') as outfile:
	with open(os.path.join(foldersave,'input_file.txt')) as infile:
		for line in infile:
			outfile.write(line)
	with open(os.path.join(foldersave,'target_file.txt')) as infile:
		for line in infile:
			outfile.write(line)
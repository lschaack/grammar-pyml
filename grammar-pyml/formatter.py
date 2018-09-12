from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.stanford import StanfordTokenizer
from collections import Counter
import os

UNK_THRESHOLD = 0 # 2

# TODO:
# take filepath arg, write path as 'path_beginning' + '.processed.txt'
if __name__ == '__main__':
	read_path = '../data/lovecraft_collected_works.txt'
	write_path = '../data/lcw_processed_nounk.txt'

	with open(os.path.abspath(read_path), encoding='utf-8') as f:
		text = f.read()

		nist = StanfordTokenizer()
		counts = Counter([token for token in nist.tokenize(text)])
		# below, +2 is for <unk> and <eos>
		print('Vocab size: {}'.format(
			len([word for word in counts if counts[word] > UNK_THRESHOLD])+1)
		)

		# Needlessly confusing, but kind of fun coming back from Haskell
		with open(os.path.abspath(write_path) , 'w') as f_write:
			# Write one sentence per line, tokens separated by spaces
			f_write.write('\n'.join(
				[' '.join(
					[token if counts[token] > UNK_THRESHOLD else '<unk>'
						for token in nist.tokenize(sent)])
					for sent in sent_tokenize(text)]
			))
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import argparse
import os

UNK_THRESHOLD = 0 # 2

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('readpath',\
		help='The relative filepath to the text to be formatted')
	parser.add_argument('-w', '--writepath', default=None,\
		help='The relative filepath to be written to')
	args = parser.parse_args()
	
	assert os.path.splitext(args.readpath)[1] == '.txt', 'Formatter only takes utf-8 encoded .txt files'
	read_path = os.path.abspath(args.readpath)
	write_path = os.path.abspath(args.writepath) # returns working directory if None
	if args.writepath is None:
		write_path = '{}_processed.txt'.format(os.path.splitext(read_path)[0])

	try: # this was missing for me, so maybe it will be for others as well
		nltk.data.find('tokenizers/punkt')
	except LookupError:
		nltk.download('punkt')

	with open(read_path, encoding='utf-8') as f:
		text = f.read()

		counts = Counter([token for token in word_tokenize(text)])
		# below, +2 is for <unk> and <eos>
		print('Vocab size: {}'.format(
			len([word for word in counts if counts[word] > UNK_THRESHOLD])+1)
		)

		# Needlessly confusing, but kind of fun coming back from Haskell
		with open(write_path, 'w') as f_write:
			# Write one sentence per line, tokens separated by spaces
			f_write.write('\n'.join(
				[' '.join(
					[token if counts[token] > UNK_THRESHOLD else '<unk>'
						for token in word_tokenize(sent)])
					for sent in sent_tokenize(text)]
			))
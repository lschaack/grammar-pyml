import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import argparse
import os

# Wrapper for file output from get_formatted_text when used from terminal/command line
def output_general(read_path, write_path, unk_threshold, splitter_callback, verbosity):
    if verbosity > 1: print('Opening file:', read_path)
    with open(read_path, encoding='utf-8') as f:
        text = f.read()
        if verbosity > 1: print('Read file contents, writing to:', write_path)
        with open(write_path, 'w', encoding='utf-8') as f_write:
            f_write.write(get_formatted_text(text, unk_threshold, splitter_callback=splitter_callback, verbosity=verbosity))
            if verbosity > 1: print('Finished')

def get_formatted_text(text, unk_threshold=0, splitter_callback=list, verbosity=0):
    counts = Counter([token for token in splitter_callback(text)])

    if verbosity > 0:
        print('Vocab size accounting for <unk> and <eos>: {}'.format(
                len([word for word in counts if counts[word] >= unk_threshold]) + 1 + int(not unk_threshold)))

    # Needlessly confusing, but kind of fun coming back from Haskell
    # Write one sentence per line, tokens separated by spaces
    formatted = '\n'.join(
        [' '.join(
            [token if counts[token] >= unk_threshold else '<unk>'
                for token in splitter_callback(sent)])
            for sent in sent_tokenize(text)])

    if verbosity > 2:
        print("Formatted text:", formatted)

    return formatted

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('readpath',\
        help='The relative filepath to the text to be formatted')
    parser.add_argument('-w', '--writepath', default=None,\
        help='The relative filepath to be written to')
    parser.add_argument('-u', '--unknownthreshold', type=int, default=0,\
        help='The minimum number of occurrences for a word to NOT be tagged <unk>')
    parser.add_argument('-m', '--mode', default='char',\
        help='The mode of output, one of "word" or "char"')
    parser.add_argument('-v', '--verbose', type=int, default=0,\
        help='How much to print, one of 1, 2, or 3')
    args = parser.parse_args()
    
    assert os.path.splitext(args.readpath)[1] == '.txt', 'Formatter only takes utf-8 encoded .txt files'
    read_path = os.path.abspath(args.readpath)
    write_path = os.path.abspath(args.writepath) # returns working directory if None
    if args.writepath is None:
        write_path = '{}_processed.txt'.format(os.path.splitext(read_path)[0])
    unk_threshold = args.unknownthreshold

    if args.mode == 'char':
        output_general(read_path, write_path, args.unknownthreshold, list, args.verbose)
    elif args.mode == 'word':
        try: # this was missing for me, so maybe it will be for others as well
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        output_general(read_path, write_path, args.unknownthreshold, word_tokenize, args.verbose)
    else:
        raise ValueError('Mode must be one of "word" or "char"')
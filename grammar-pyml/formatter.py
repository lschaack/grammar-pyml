from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import os

UNK_THRESHOLD = 2

if __name__ == '__main__':
    data_path = '../data/lovecraft_collected_works.txt'
    write_path = '../data/lcw_processed.txt'

    with open(os.path.abspath(data_path)) as f:
        text = f.read()

        counts = Counter([token for token in word_tokenize(text)])
        # below, +1 is for <unk>
        print('Vocab size: {}'.format(
            len([word for word in counts if counts[word] > UNK_THRESHOLD])+1)
        )

        # Needlessly confusing, but kind of fun coming back from Haskell
        with open(os.path.abspath(write_path) , 'w') as f_write:
            # Write one sentence per line, tokens separated by spaces
            f_write.write('\n'.join(
                [' '.join(
                    [token if counts[token] > UNK_THRESHOLD else '<unk>'
                        for token in word_tokenize(sent)])
                    for sent in sent_tokenize(text)]
            ))
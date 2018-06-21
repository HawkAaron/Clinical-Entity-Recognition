import os
import argparse

from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word

parser = argparse.ArgumentParser(description='NER config')
parser.add_argument('--train_embedding', default=False, action='store_true')
parser.add_argument('--epochs', type=int ,default=50)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=20)
# lr
parser.add_argument('--lr_method', type=str, default='adam')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lr_decay', type=float, default=0.5)
parser.add_argument('--clip', type=float, default=-1)
parser.add_argument('--no_improve', type=int, default=5)
# embed
parser.add_argument('--dim_word', type=int, default=300)
parser.add_argument('--dim_char', type=int, default=100)
parser.add_argument('--dim_cap', type=int, default=5)
# hidden
parser.add_argument('--hidden_word', type=int, default=300)
parser.add_argument('--hidden_char', type=int, default=100)
parser.add_argument('--layers', type=int, default=1)
# lstm
parser.add_argument('--word_mode', type=str, default='LSTM')
parser.add_argument('--num_proj', type=int, default=0)
parser.add_argument('--peephole', default=False, action='store_true')
parser.add_argument('--char_peephole', default=False, action='store_true')
# char
parser.add_argument('--use_char', default=False, action='store_true')
parser.add_argument('--char_mode', type=str, default='LSTM')
# cap
parser.add_argument('--use_cap', default=False, action='store_true')
# pretrained word embeddings
parser.add_argument('--use_pretrained', default=False, action='store_true')
# dir
parser.add_argument('--dir', default='exp/test')
args = parser.parse_args()

class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        os.makedirs(self.dir_output, exist_ok=True)
 
        # save args
        with open(os.path.join(self.dir_output, 'args'), 'w') as f:
            f.write(str(args))

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words, self.words_idx = load_vocab(self.filename_words)
        self.vocab_tags, self.tags_idx  = load_vocab(self.filename_tags)
        self.vocab_chars, self.chars_idx = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars, caps=self.use_caps)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)


    # configs
    train_embeddings = args.train_embedding
    nepochs = args.epochs
    dropout = args.dropout
    batch_size = args.batch_size
    lr_method = args.lr_method
    lr = args.lr
    lr_decay = args.lr_decay
    clip = args.clip
    nepoch_no_imprv = args.no_improve

    dim_word = args.dim_word
    dim_char = args.dim_char
    dim_cap = args.dim_cap

    hidden_size_lstm = args.hidden_word # lstm on word embeddings
    hidden_size_char = args.hidden_char # lstm on chars
    layers = args.layers
    word_mode = args.word_mode # word embeddings lstm type
    num_proj = args.num_proj # lstm projection
    peephole = args.peephole # peephole lstm
    char_peephole = args.char_peephole # char peephole lstm

    use_chars = args.use_char
    char_mode = args.char_mode # char embeddings lstm type
    use_caps = args.use_cap
    use_pretrained = args.use_pretrained
    use_crf = True

    # glove files
    filename_glove = 'data/glove.6B/glove.6B.{}d.txt'.format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = 'data/glove.6B.{}d.trimmed.npz'.format(dim_word)
    
    # dataset
    filename_train = 'data/trdv.txt'
    filename_dev = 'data/dev.txt'
    filename_test = 'data/test.txt'

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from datset with build_data.py)
    filename_words = 'data/words.txt'
    filename_tags = 'data/tags.txt'
    filename_chars = 'data/chars.txt'

    # out dir
    dir_output = args.dir
    dir_model = os.path.join(args.dir, 'params')
    path_log = os.path.join(dir_output, 'log.txt')

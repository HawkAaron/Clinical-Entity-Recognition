from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.ensemble import Ensemble
from model.debug import Debug
from model.config import Config


def main():
    # create instance of config
    config = Config()

    # build model
    if config.ensembles:
        model = Debug(config) if config.debug else Ensemble(config)
    else:
        model = NERModel(config)
    model.build()
    # debug model
    if config.debug: model.restore_session(config.ensembles[0]+'/params')

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)

if __name__ == "__main__":
    main()

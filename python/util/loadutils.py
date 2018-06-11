import fasttext
import json

from constants.paths import Paths
from util.readers import GenderDataReader, DictionaryReader


def load_json_from_path(path):
    with open(path, "r") as fin:
        return json.load(fin)


def load_word_vecs():
    return fasttext.load_model(Paths.Resources.Fasttext50d)


def load_gender_data():
    w2g_in = open(Paths.Resources.GenderData, 'rb')
    return GenderDataReader.load(w2g_in, True, True)


def load_animate_data():
    ani_in = open(Paths.Resources.AnimateUnigram, 'rb')
    return DictionaryReader.load_string_set(ani_in)


def load_inanimate_data():
    ina_in = open(Paths.Resources.InanimateUnigram, 'rb')
    return DictionaryReader.load_string_set(ina_in)

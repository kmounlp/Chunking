__author__ = 'max'


class Sentence(object):
    def __init__(self, cont, cont_ids, func, func_ids, char_c_seqs, char_c_id_seqs, char_f_seqs, char_f_id_seqs, sent_id, sentence):
        self.cont = cont
        self.cont_ids = cont_ids
        self.func = func
        self.func_ids = func_ids
        self.char_c_seqs = char_c_seqs
        self.char_c_id_seqs = char_c_id_seqs
        self.char_f_seqs = char_f_seqs
        self.char_f_id_seqs = char_f_id_seqs
        self.sent_id = sent_id
        self.sentence = sentence

    def length(self):
        return len(self.cont)

    def get_sent_id(self):
        return self.sent_id

    def get_sentence(self):
        return self.sentence


class DependencyInstance(object):
    def __init__(self, sentence, postags_c, pos_c_ids, postags_f, pos_f_ids, heads, types, type_ids):
        self.sentence = sentence
        self.postags_c = postags_c
        self.pos_c_ids = pos_c_ids
        self.postags_f = postags_f
        self.pos_f_ids = pos_f_ids
        self.heads = heads
        self.types = types
        self.type_ids = type_ids

    def length(self):
        return self.sentence.length()


class NERInstance(object):
    def __init__(self, sentence, postags, pos_ids, chunk_tags, chunk_ids, ner_tags, ner_ids):
        self.sentence = sentence
        self.postags = postags
        self.pos_ids = pos_ids
        self.chunk_tags = chunk_tags
        self.chunk_ids = chunk_ids
        self.ner_tags = ner_tags
        self.ner_ids = ner_ids

    def length(self):
        return self.sentence.length()

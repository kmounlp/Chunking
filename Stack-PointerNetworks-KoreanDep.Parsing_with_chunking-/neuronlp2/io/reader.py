__author__ = 'max'

from .instance import DependencyInstance, NERInstance
from .instance import Sentence
from .conllx_data import ROOT, ROOT_POS, ROOT_CHAR, ROOT_TYPE, END, END_POS, END_CHAR, END_TYPE
from . import utils


class CoNLLXReader(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        self.__source_file = open(file_path, 'r', encoding='utf-8')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True, symbolic_root=False, symbolic_end=False, sent_id=0):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return

        lines = []
        while len(line.strip()) > 0:
            if line.strip().startswith("#"):
                line = self.__source_file.readline()
            else:
                line = line.strip()
                lines.append(line.split('\t'))
                line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        cont_seqs = []
        cont_id_seqs = []
        func_seqs = []
        func_id_seqs = []
        char_c_seqs = []
        char_c_id_seqs = []
        char_f_seqs = []
        char_f_id_seqs = []
        pos_c_seqs = []
        pos_c_id_seqs = []
        pos_f_seqs = []
        pos_f_id_seqs = []
        types = []
        type_ids = []
        heads = []

        if symbolic_root:
            cont_seqs.append([ROOT, ])
            cont_id_seqs.append([self.__word_alphabet.get_index(ROOT), ])
            func_seqs.append([ROOT, ])
            func_id_seqs.append([self.__word_alphabet.get_index(ROOT), ])
            char_c_seqs.append([[ROOT_CHAR, ]])
            char_c_id_seqs.append([[self.__char_alphabet.get_index(ROOT_CHAR), ]])
            char_f_seqs.append([[ROOT_CHAR, ]])
            char_f_id_seqs.append([[self.__char_alphabet.get_index(ROOT_CHAR), ]])
            pos_c_seqs.append([ROOT_POS, ])
            pos_c_id_seqs.append([self.__pos_alphabet.get_index(ROOT_POS), ])
            pos_f_seqs.append([ROOT_POS, ])
            pos_f_id_seqs.append([self.__pos_alphabet.get_index(ROOT_POS), ])
            types.append(ROOT_TYPE)
            type_ids.append(self.__type_alphabet.get_index(ROOT_TYPE))
            heads.append(0)

        for tokens in lines:
            ###########################################################################################
            # word
            cont = []
            cont_ids = []
            func = []
            func_ids = []
            chars_c = []
            char_c_ids = []
            chars_f = []
            char_f_ids = []

            cont_word = tokens[1].split("_")
            func_word = tokens[2].split("_")

            # contents words
            for w_cont in cont_word:
                w_cont_ = utils.DIGIT_RE.sub("0", w_cont) if normalize_digits else w_cont
                cont.append(w_cont_)
                cont_ids.append(self.__word_alphabet.get_index(w_cont_))
                w_conts = []
                w_cont_ids = []
                for char in w_cont:
                    w_conts.append(char)
                    w_cont_ids.append(self.__char_alphabet.get_index(char))
                if len(chars_c) > utils.MAX_CHAR_LENGTH:
                    w_conts = w_conts[:utils.MAX_CHAR_LENGTH]
                    w_cont_ids = w_cont_ids[:utils.MAX_CHAR_LENGTH]
                chars_c.append(w_conts)
                char_c_ids.append(w_cont_ids)
            if len(cont) > utils.MAX_EOJUL_LENGTH:
                cont = cont[utils.MAX_EOJUL_LENGTH]
                cont_ids = cont_ids[:utils.MAX_EOJUL_LENGTH]

            # functional words
            for w_func in func_word:
                w_func_ = utils.DIGIT_RE.sub("0", w_func) if normalize_digits else w_func
                func.append(w_func_)
                func_ids.append(self.__word_alphabet.get_index(w_func_))
                w_funcs = []
                w_func_ids = []
                for char in w_func:
                    w_funcs.append(char)
                    w_func_ids.append(self.__char_alphabet.get_index(char))
                if len(chars_f) > utils.MAX_CHAR_LENGTH:
                    w_funcs = w_funcs[:utils.MAX_CHAR_LENGTH]
                    w_func_ids = w_func_ids[:utils.MAX_CHAR_LENGTH]
                chars_f.append(w_funcs)
                char_f_ids.append(w_func_ids)
            if len(func) > utils.MAX_EOJUL_LENGTH:
                func = func[:utils.MAX_EOJUL_LENGTH]
                func_ids = func_ids[:utils.MAX_EOJUL_LENGTH]

            cont_seqs.append(cont)
            cont_id_seqs.append(cont_ids)
            func_seqs.append(func)
            func_id_seqs.append(func_ids)
            char_c_seqs.append(chars_c)
            char_c_id_seqs.append(char_c_ids)
            char_f_seqs.append(chars_f)
            char_f_id_seqs.append(char_f_ids)
            ###########################################################################################
            # pos
            poss_c = []
            pos_c_ids = []
            poss_f = []
            pos_f_ids = []

            pos_tag = tokens[5].split("+")
            pos_cont = pos_tag[:len(cont_word)]
            pos_func = pos_tag[len(cont_word):]

            for pos in pos_cont:
                poss_c.append(pos)
                pos_c_ids.append(self.__pos_alphabet.get_index(pos))
            if len(poss_c) > utils.MAX_POS_LENGTH:
                poss_c = poss_c[:utils.MAX_POS_LENGTH]
                pos_c_ids = pos_c_ids[:utils.MAX_POS_LENGTH]

            if pos_func:
                for pos in pos_func:
                    poss_f.append(pos)
                    pos_f_ids.append(self.__pos_alphabet.get_index(pos))
                if len(poss_f) > utils.MAX_POS_LENGTH:
                    poss_f = poss_f[:utils.MAX_POS_LENGTH]
                    pos_f_ids = pos_f_ids[:utils.MAX_POS_LENGTH]
            else:
                poss_f.append('-')
                pos_f_ids.append(self.__pos_alphabet.get_index('-'))
                if len(poss_f) > utils.MAX_POS_LENGTH:
                    poss_f = poss_f[:utils.MAX_POS_LENGTH]
                    pos_f_ids = pos_f_ids[:utils.MAX_POS_LENGTH]


            pos_c_seqs.append(poss_c)
            pos_c_id_seqs.append(pos_c_ids)
            pos_f_seqs.append(poss_f)
            pos_f_id_seqs.append(pos_f_ids)
            ###########################################################################################

            #word = utils.DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
            #pos = tokens[4]
            head = int(tokens[7])  # 191117
            type = tokens[8]  # 191117

            #words.append(word)
            #word_ids.append(self.__word_alphabet.get_index(word))

            #postags.append(pos)
            #pos_ids.append(self.__pos_alphabet.get_index(pos))

            types.append(type)
            type_ids.append(self.__type_alphabet.get_index(type))

            heads.append(head)

        if symbolic_end:
            cont_seqs.append([END, ])
            cont_id_seqs.append([self.__word_alphabet.get_index(END), ])
            func_seqs.append([END, ])
            func_id_seqs.append([self.__word_alphabet.get_index(END), ])
            char_c_seqs.append([END_CHAR, ])
            char_c_id_seqs.append([self.__char_alphabet.get_index(END_CHAR), ])
            char_f_seqs.append([END_CHAR, ])
            char_f_id_seqs.append([self.__char_alphabet.get_index(END_CHAR), ])
            pos_c_seqs.append([END_POS, ])
            pos_c_id_seqs.append([self.__pos_alphabet.get_index(END_POS), ])
            pos_f_seqs.append([END_POS, ])
            pos_f_id_seqs.append([self.__pos_alphabet.get_index(END_POS), ])
            types.append(END_TYPE)
            type_ids.append(self.__type_alphabet.get_index(END_TYPE))
            heads.append(0)

        return DependencyInstance(\
            Sentence(cont_seqs, cont_id_seqs, func_seqs, func_id_seqs, char_c_seqs, char_c_id_seqs, char_f_seqs, char_f_id_seqs, sent_id, lines), \
            pos_c_seqs, pos_c_id_seqs, pos_f_seqs, pos_f_id_seqs, heads, types, type_ids)


class CoNLL03Reader(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__chunk_alphabet = chunk_alphabet
        self.__ner_alphabet = ner_alphabet

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            line = line.decode('utf-8')
            lines.append(line.split(' '))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        chunk_tags = []
        chunk_ids = []
        ner_tags = []
        ner_ids = []

        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > utils.MAX_CHAR_LENGTH:
                chars = chars[:utils.MAX_CHAR_LENGTH]
                char_ids = char_ids[:utils.MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            word = utils.DIGIT_RE.sub(b"0", tokens[1]) if normalize_digits else tokens[1]
            pos = tokens[2]
            chunk = tokens[3]
            ner = tokens[4]

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            chunk_tags.append(chunk)
            chunk_ids.append(self.__chunk_alphabet.get_index(chunk))

            ner_tags.append(ner)
            ner_ids.append(self.__ner_alphabet.get_index(ner))

        return NERInstance(Sentence(words, word_ids, char_seqs, char_id_seqs), postags, pos_ids, chunk_tags, chunk_ids,
                           ner_tags, ner_ids)

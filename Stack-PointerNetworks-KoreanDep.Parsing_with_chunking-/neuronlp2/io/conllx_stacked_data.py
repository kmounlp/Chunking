__author__ = 'max'

import numpy as np
import torch
from torch.autograd import Variable
from .conllx_data import _buckets, PAD_ID_WORD, PAD_ID_CHAR, PAD_ID_TAG, UNK_ID
from .conllx_data import NUM_SYMBOLIC_TAGS
from .conllx_data import create_alphabets, load_alphabets
from . import utils
from .reader import CoNLLXReader


def _obtain_child_index_for_left2right(heads):
    child_ids = [[] for _ in range(len(heads))]
    # skip the symbolic root.
    for child in range(1, len(heads)):
        head = heads[child]
        child_ids[head].append(child)
    return child_ids


def _obtain_child_index_for_inside_out(heads):
    child_ids = [[] for _ in range(len(heads))]
    for head in range(len(heads)):
        # first find left children inside-out
        for child in reversed(range(1, head)):
            if heads[child] == head:
                child_ids[head].append(child)
        # second find right children inside-out
        for child in range(head + 1, len(heads)):
            if heads[child] == head:
                child_ids[head].append(child)
    return child_ids


def _obtain_child_index_for_depth(heads, reverse):
    def calc_depth(head):
        children = child_ids[head]
        max_depth = 0
        for child in children:
            depth = calc_depth(child)
            child_with_depth[head].append((child, depth))
            max_depth = max(max_depth, depth + 1)
        child_with_depth[head] = sorted(child_with_depth[head], key=lambda x: x[1], reverse=reverse)
        return max_depth

    child_ids = _obtain_child_index_for_left2right(heads)
    child_with_depth = [[] for _ in range(len(heads))]
    calc_depth(0)
    return [[child for child, depth in child_with_depth[head]] for head in range(len(heads))]


def _generate_stack_inputs(heads, types, prior_order):
    if prior_order == 'deep_first':
        child_ids = _obtain_child_index_for_depth(heads, True)
    elif prior_order == 'shallow_first':
        child_ids = _obtain_child_index_for_depth(heads, False)
    elif prior_order == 'left2right':
        child_ids = _obtain_child_index_for_left2right(heads)
    elif prior_order == 'inside_out':
        child_ids = _obtain_child_index_for_inside_out(heads)
    else:
        raise ValueError('Unknown prior order: %s' % prior_order)

    stacked_heads = []
    children = []
    siblings = []
    stacked_types = []
    skip_connect = []
    prev = [0 for _ in range(len(heads))]
    sibs = [0 for _ in range(len(heads))]
    stack = [0]
    position = 1
    while len(stack) > 0:
        head = stack[-1]
        stacked_heads.append(head)
        siblings.append(sibs[head])
        child_id = child_ids[head]
        skip_connect.append(prev[head])
        prev[head] = position
        if len(child_id) == 0:
            children.append(head)
            sibs[head] = 0
            stacked_types.append(PAD_ID_TAG)
            stack.pop()
        else:
            child = child_id.pop(0)
            children.append(child)
            sibs[head] = child
            stack.append(child)
            stacked_types.append(types[child])
        position += 1

    return stacked_heads, children, siblings, stacked_types, skip_connect


def read_stacked_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_size=None, normalize_digits=True, prior_order='deep_first'):
    data = [[] for _ in _buckets]
    max_lemma_length = [0 for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size <= bucket_size:
                stacked_heads, children, siblings, stacked_types, skip_connect = _generate_stack_inputs(inst.heads, inst.type_ids, prior_order)
                data[bucket_id].append([sent.cont_ids, sent.func_ids, sent.char_c_id_seqs, sent.char_f_id_seqs, inst.pos_c_ids, inst.pos_f_ids, inst.heads, inst.type_ids, stacked_heads, children, siblings, stacked_types, skip_connect, sent.sentence])  # 191117
                # char_c_lengths = []
                # char_f_lengths = []
                # print("=====chech char_seqs=====")
                # print("sent.char_c_seqs: ", sent.char_c_seqs)
                # for eojul in sent.char_c_seqs:
                #     print("c_eojul: ", eojul)
                #     for char_seq in eojul:
                #         print("c_char_seq: ", char_seq)
                #         char_c_lengths.append(len(char_seq))
                # max_c_len = max(char_c_lengths)
                # print("max_c_len: ", max_c_len)
                #
                # print("sent.char_f_seqs: ", sent.char_f_seqs)
                # for eojul in sent.char_f_seqs:
                #     print("f_eojul: ", eojul)
                #     for char_seq in eojul:
                #         print("c_char_seq: ", char_seq)
                #         char_f_lengths.append(len(char_seq))
                # max_f_len = max(char_f_lengths)
                # print("max_f_len: ", max_f_len)

                max_c_len = max([len(char_seq) for eojul in sent.char_c_seqs for char_seq in eojul])
                max_f_len = max([len(char_seq) for eojul in sent.char_f_seqs for char_seq in eojul])
                max_len = max(max_c_len, max_f_len)
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len

                max_c_len = max([len(word_seq) for word_seq in sent.cont])
                max_f_len = max([len(word_seq) for word_seq in sent.func])
                max_len = max(max_c_len, max_f_len)
                if max_lemma_length[bucket_id] < max_len:
                    max_lemma_length[bucket_id] = max_len

                break
        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    reader.close()
    print("Total number of data: %d" % counter)
    return data, max_lemma_length, max_char_length


def read_stacked_data_to_variable(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                  max_size=None, normalize_digits=True, prior_order='deep_first', use_gpu=False):
    data, max_lemma_length, max_char_length = read_stacked_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_size=max_size, normalize_digits=normalize_digits, prior_order=prior_order)
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]

    data_variable = []

    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_variable.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        lemma_length = min(utils.MAX_EOJUL_LENGTH, max_lemma_length[bucket_id] + utils.NUM_EOJUL_PAD)
        char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
        cid_inputs = np.empty([bucket_size, bucket_length, lemma_length], dtype=np.int64)  # 191118
        fid_inputs = np.empty([bucket_size, bucket_length, lemma_length], dtype=np.int64)  # 191118
        ccid_inputs = np.empty([bucket_size, bucket_length, lemma_length, char_length], dtype=np.int64)  # 191118
        cfid_inputs = np.empty([bucket_size, bucket_length, lemma_length, char_length], dtype=np.int64)  # 191118
        pcid_inputs = np.empty([bucket_size, bucket_length, lemma_length], dtype=np.int64)  # 191118
        pfid_inputs = np.empty([bucket_size, bucket_length, lemma_length], dtype=np.int64)  # 191118
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks_e = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single_c = np.zeros([bucket_size, bucket_length, lemma_length], dtype=np.int64)
        single_f = np.zeros([bucket_size, bucket_length, lemma_length], dtype=np.int64)
        lengths_e = np.empty(bucket_size, dtype=np.int64)

        stack_hid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        chid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        ssid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        stack_tid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        skip_connect_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)

        masks_d = np.zeros([bucket_size, 2 * bucket_length - 1], dtype=np.float32)
        lengths_d = np.empty(bucket_size, dtype=np.int64)

        sentences = []

        for i, inst in enumerate(data[bucket_id]):
            cids, fids, ccid_seqs, cfid_seqs, pcids, pfids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids, sentence = inst
            # wids, cid_seqs, pids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids, sentence = inst
            inst_size = max(len(cids), len(fids))
            # print(len(cids), len(fids))
            lengths_e[i] = inst_size
            # [ word ids ]
            # cont ids
            for w, w_ids in enumerate(cids):
                cid_inputs[i, w, :len(w_ids)] = w_ids
                cid_inputs[i, w, len(w_ids):] = PAD_ID_WORD
            cid_inputs[i, inst_size:, :] = PAD_ID_WORD
            # func ids
            for w, w_ids in enumerate(fids):
                fid_inputs[i, w, :len(w_ids)] = w_ids
                fid_inputs[i, w, len(w_ids):] = PAD_ID_WORD
            fid_inputs[i, inst_size:, :] = PAD_ID_WORD

            # [ char ids ]
            # cont char ids
            # print(sentence)
            for c, cids in enumerate(ccid_seqs):
                for l, lids in enumerate(cids):
                    ccid_inputs[i, c, l, :len(lids)] = lids
                    ccid_inputs[i, c, l, len(lids):] = PAD_ID_CHAR
                ccid_inputs[i, c, len(cids):, :] = PAD_ID_CHAR
            ccid_inputs[i, inst_size:, :, :] = PAD_ID_CHAR
            # func char ids
            for c, cids in enumerate(cfid_seqs):
                for l, lids in enumerate(cids):
                    cfid_inputs[i, c, l, :len(lids)] = lids
                    cfid_inputs[i, c, l, len(lids):] = PAD_ID_CHAR
                cfid_inputs[i, c, len(cids):, :] = PAD_ID_CHAR
            cfid_inputs[i, inst_size:, :, :] = PAD_ID_CHAR

            # [ pos ids ]
            # cont pos ids
            for p, p_ids in enumerate(pcids):
                pcid_inputs[i, p, :len(p_ids)] = p_ids
                pcid_inputs[i, p, len(p_ids):] = PAD_ID_TAG
            pcid_inputs[i, inst_size:, :] = PAD_ID_TAG
            # func pos ids
            for p, p_ids in enumerate(pfids):
                pfid_inputs[i, p, :len(p_ids)] = p_ids
                pfid_inputs[i, p, len(p_ids):] = PAD_ID_TAG
            pfid_inputs[i, inst_size:, :] = PAD_ID_TAG

            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks_e  # role?
            masks_e[i, :inst_size] = 1.0
            for j, lids in enumerate(cids):
                for k, wid in enumerate(lids):
                    if word_alphabet.is_singleton(wid):
                        single_c[i, j, k] = 1
            for j, lids in enumerate(fids):
                for k, wid in enumerate(lids):
                    if word_alphabet.is_singleton(wid):
                        single_f[i, j, k] = 1

            inst_size_decoder = 2 * inst_size - 1
            lengths_d[i] = inst_size_decoder
            # stacked heads
            # print("=====stack_hids=====")
            # print("shape: ", stack_hid_inputs.shape)
            # print("size: ", stack_hid_inputs.size)
            # print("stack_hids: ", stack_hids)
            # print("len(stack_hids): ", len(stack_hids))
            # print("stack_hid_inputs: ", stack_hid_inputs)
            # print("inst_size_decoder: ", inst_size_decoder)
            stack_hid_inputs[i, :inst_size_decoder] = stack_hids  # [bucket_size, 2 * bucket_length - 1]
            stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # children
            chid_inputs[i, :inst_size_decoder] = chids
            chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # siblings
            ssid_inputs[i, :inst_size_decoder] = ssids
            ssid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # stacked types
            stack_tid_inputs[i, :inst_size_decoder] = stack_tids
            stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # skip connects
            skip_connect_inputs[i, :inst_size_decoder] = skip_ids
            skip_connect_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # masks_d
            masks_d[i, :inst_size_decoder] = 1.0
            sentences.append(sentence)

        words_c = torch.from_numpy(cid_inputs)
        words_f = torch.from_numpy(fid_inputs)
        chars_c = torch.from_numpy(ccid_inputs)
        chars_f = torch.from_numpy(cfid_inputs)
        pos_c = torch.from_numpy(pcid_inputs)
        pos_f = torch.from_numpy(pfid_inputs)
        heads = torch.from_numpy(hid_inputs)
        types = torch.from_numpy(tid_inputs)
        masks_e = torch.from_numpy(masks_e)
        single_c = torch.from_numpy(single_c)
        single_f = torch.from_numpy(single_f)
        lengths_e = torch.from_numpy(lengths_e)

        stacked_heads = torch.from_numpy(stack_hid_inputs)
        children = torch.from_numpy(chid_inputs)
        siblings = torch.from_numpy(ssid_inputs)
        stacked_types = torch.from_numpy(stack_tid_inputs)
        skip_connect = torch.from_numpy(skip_connect_inputs)
        masks_d = torch.from_numpy(masks_d)
        lengths_d = torch.from_numpy(lengths_d)

        if False:
            words_c = words_c.cuda()
            words_f = words_f.cuda()
            chars_c = chars_c.cuda()
            chars_f = chars_f.cuda()
            pos_c = pos_c.cuda()
            pos_f = pos_f.cuda()
            heads = heads.cuda()
            types = types.cuda()
            masks_e = masks_e.cuda()
            single_c = single_c.cuda()
            single_f = single_f.cuda()
            lengths_e = lengths_e.cuda()
            stacked_heads = stacked_heads.cuda()
            children = children.cuda()
            siblings = siblings.cuda()
            stacked_types = stacked_types.cuda()
            skip_connect = skip_connect.cuda()
            masks_d = masks_d.cuda()
            lengths_d = lengths_d.cuda()

        data_variable.append(((words_c, words_f, chars_c, chars_f, pos_c, pos_f, heads, types, masks_e, single_c, single_f, lengths_e),
                              (stacked_heads, children, siblings, stacked_types, skip_connect, masks_d, lengths_d),
                              sentences))

    return data_variable, bucket_sizes


def get_batch_stacked_variable(data, batch_size, unk_replace=0., use_gpu=False):
    data_variable, bucket_sizes = data
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
    bucket_length = _buckets[bucket_id]

    data_encoder, data_decoder, _ = data_variable[bucket_id]
    words_c, words_f, chars_c, chars_f, pos_c, pos_f, heads, types, masks_e, single_c, single_f, lengths_e = data_encoder  # 191118
    stacked_heads, children, siblings, stacked_types, skip_connect, masks_d, lengths_d = data_decoder
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)
    index = torch.randperm(bucket_size).long()[:batch_size]
    # if words.is_cuda:
    #     index = index.cuda()
    lemma_length_c = words_c.size(2)  # unk_replace  # 아마 같을 것
    words_c = words_c[index]  # unk_replace
    lemma_length_f = words_f.size(2)  # unk_replace  # 아마 같을 것
    words_f = words_f[index]  # unk_replace

    if unk_replace:
        ones = torch.LongTensor(single_c.data.new(batch_size, bucket_length, lemma_length_c).fill_(1))
        noise = torch.LongTensor(masks_e.data.new(batch_size, bucket_length, lemma_length_c).bernoulli_(unk_replace).long())
        words_c = words_c * (ones - single_c[index] * noise)
        ones = torch.LongTensor(single_f.data.new(batch_size, bucket_length, lemma_length_f).fill_(1))
        noise = torch.LongTensor(masks_e.data.new(batch_size, bucket_length, lemma_length_f).bernoulli_(unk_replace).long())
        words_f = words_f * (ones - single_f[index] * noise)

    if use_gpu:
        words_c = words_c.cuda()
        words_f = words_f.cuda()
        chars_c = chars_c.cuda()
        chars_f = chars_f.cuda()
        pos_c = pos_c.cuda()
        pos_f = pos_f.cuda()
        heads = heads.cuda()
        types = types.cuda()
        masks_e = masks_e.cuda()
        lengths_e = lengths_e.cuda()
        stacked_heads = stacked_heads.cuda()
        children = children.cuda()
        siblings = siblings.cuda()
        stacked_types = stacked_types.cuda()
        skip_connect = skip_connect.cuda()
        masks_d = masks_d.cuda()
        lengths_d = lengths_d.cuda()

    return (words_c, words_f, chars_c[index], chars_f[index], pos_c[index], pos_f[index], heads[index], types[index], masks_e[index], lengths_e[index]), \
           (stacked_heads[index], children[index], siblings[index], stacked_types[index], skip_connect[index], masks_d[index], lengths_d[index])


def iterate_batch_stacked_variable(data, batch_size, unk_replace=0., shuffle=False, use_gpu=False):
    data_variable, bucket_sizes = data

    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]
        if bucket_size == 0:
            continue
        data_encoder, data_decoder, data_sentences = data_variable[bucket_id]
        words_c, words_f, chars_c, chars_f, pos_c, pos_f, heads, types, masks_e, single_c, single_f, lengths_e = data_encoder
        stacked_heads, children, siblings, stacked_types, skip_connect, masks_d, lengths_d = data_decoder
        # lemma_length = words.size(2)
        lemma_length_c = words_c.size(2)  # unk_replace  # 191118
        lemma_length_f = words_f.size(2)  # unk_replace  # 191118

        if unk_replace:
            ones = Variable(single_c.data.new(bucket_size, bucket_length, lemma_length_c).fill_(1))
            noise = Variable(masks_e.data.new(bucket_size, bucket_length, lemma_length_c).bernoulli_(unk_replace).long())
            words_c = words_c * (ones - single_c * noise)
            ones = Variable(single_f.data.new(bucket_size, bucket_length, lemma_length_f).fill_(1))
            noise = Variable(masks_e.data.new(bucket_size, bucket_length, lemma_length_f).bernoulli_(unk_replace).long())
            words_f = words_f * (ones - single_f * noise)


        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            if words_c.is_cuda:
                indices = indices.cuda()
            if words_f.is_cuda:
                indices = indices.cuda()
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            if use_gpu:
                yield (words_c[excerpt].cuda(), words_f[excerpt].cuda(), chars_c[excerpt].cuda(), chars_f[excerpt].cuda(), pos_c[excerpt].cuda(), pos_f[excerpt].cuda(), heads[excerpt].cuda(), types[excerpt].cuda(), masks_e[excerpt].cuda(), lengths_e[excerpt].cuda()), \
                      (stacked_heads[excerpt].cuda(), children[excerpt].cuda(), siblings[excerpt].cuda(), stacked_types[excerpt].cuda(), skip_connect[excerpt].cuda(), masks_d[excerpt].cuda(), lengths_d[excerpt].cuda()), \
                      data_sentences[excerpt]
            else:
                yield (words_c[excerpt], words_f[excerpt], chars_c[excerpt], chars_f[excerpt], pos_c[excerpt], pos_f[excerpt], heads[excerpt], types[excerpt], masks_e[excerpt], lengths_e[excerpt]), \
                      (stacked_heads[excerpt], children[excerpt], siblings[excerpt], stacked_types[excerpt], skip_connect[excerpt], masks_d[excerpt], lengths_d[excerpt]), \
                      data_sentences[excerpt]

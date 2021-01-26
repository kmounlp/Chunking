import pickle
from copy import deepcopy
from collections import namedtuple
from pprint import pprint


# def convertv2s(idx):
#     with open("tag_enc.pickle", "rb") as f:
#         tag_emb = pickle.load(f)
#     label = []
#     for sent in idx:
#         sent_label = []
#         for num in sent:
#             sent_label.extend([tag for tag, vec in tag_emb.items() if num == vec])
#         label.append(sent_label)
#     return label  # label화 된 문장들 list, 2D


def compute(true, pred):
    true = set(true)
    pred = set(pred)
    tp = len(true.intersection(pred))
    fn = len(true.difference(pred))
    fp = len(pred.difference(true))
    print("precision: ", tp / (tp + fp))
    print("recall: ", tp / (tp + fn))


def collect_named_entities(tokens):
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.
    :param tokens: a list of labels
    :return: a list of Entity named-tuples
    """
    Entity = namedtuple("Entity", "e_type start_offset end_offset")

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, tag in enumerate(tokens):

        token_tag = tag

        if token_tag == 'O':
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(Entity(ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset

        elif ent_type != token_tag[2:]:
            end_offset = offset - 1
            named_entities.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token
    if ent_type and start_offset and end_offset is None:
        named_entities.append(Entity(ent_type, start_offset, offset))

    return named_entities


# def read_tags():
#     with open("tag_enc.pickle", 'rb') as fin:
#         tag_enc = pickle.load(fin)
#
#     tag_list = []
#     for tag in tag_enc:
#         tag = tag[2:]
#         if tag:
#             tag_list.append(tag)
#
#     tag_set = list(set(tag_list))
#     return tag_set


def compute_metrics(true_named_entities, pred_named_entities, chunk_set):

    eval_metrics = {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurius': 0}

    # print("chunk_set: ", chunk_set)
    # chunk_set = {'B-NX', 'B-SYX', 'I-CX', 'B-JUX', 'I-JMX', 'I-SYX', 'B-ETX', 'I-MX', 'B-JKX', 'B-CX', 'B-JCX', 'B-AX', 'B-JVX', 'I-ECX', 'B-ECX', 'B-JMX', 'B-MX', 'B-EPX', 'B-PUX', 'I-JUX', 'B-IX', 'I-NX', 'I-IX', 'B-EFX', 'I-PX', 'B-NAX', 'B-PX', 'I-ETX', 'I-EPX', 'I-PUX', 'I-JKX', 'I-JCX', 'I-AX'}
    BI_rm_chunk_list = []  # BI-tag removed
    for ch in chunk_set:
        BI_rm_chunk_list.append(ch[2:])

    # print("target_tags_no_schema: ", BI_rm_chunk_list)
    # target_tags_no_schema = ['NX', 'SYX', 'CX', 'JUX', 'JMX', 'SYX', 'ETX', 'MX', 'JKX', 'CX', 'JCX', 'AX', 'JVX', 'ECX', 'ECX', 'JMX', 'MX', 'EPX', 'PUX', 'JUX', 'IX', 'NX', 'IX', 'EFX', 'PX', 'NAX', 'PX', 'ETX', 'EPX', 'PUX', 'JKX', 'JCX', 'AX']
    target_tags_no_schema = BI_rm_chunk_list


    # overall results
    evaluation = {'strict': deepcopy(eval_metrics), 'ent_type': deepcopy(eval_metrics)}

    # results by entity type
    evaluation_agg_entities_type = {e: deepcopy(evaluation) for e in target_tags_no_schema}

    true_which_overlapped_with_pred = []  # keep track of entities that overlapped

    # print("pred_named_entities: ", pred_named_entities)
    # go through each predicted named-entity
    for pred in pred_named_entities:
        found_overlap = False

        # check if there's an exact match, i.e.: boundary and entity type match
        if pred in true_named_entities:
            true_which_overlapped_with_pred.append(pred)
            evaluation['strict']['correct'] += 1
            evaluation['ent_type']['correct'] += 1

            # print("evaluation_agg_entities_type(dictionary): ", evaluation_agg_entities_type)
            # print("pred.e_type: ", pred.e_type)
            # for the agg. by e_type results
            evaluation_agg_entities_type[pred.e_type]['strict']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['ent_type']['correct'] += 1

        else:

            # check for overlaps with any of the true entities
            for true in true_named_entities:

                # check for an exact boundary match but with a different e_type
                if true.start_offset <= pred.end_offset and pred.start_offset <= true.end_offset and true.e_type != pred.e_type:

                    # overall results
                    evaluation['strict']['incorrect'] += 1
                    evaluation['ent_type']['incorrect'] += 1

                    # aggregated by entity type results
                    evaluation_agg_entities_type[pred.e_type]['strict']['incorrect'] += 1
                    evaluation_agg_entities_type[pred.e_type]['ent_type']['incorrect'] += 1

                    true_which_overlapped_with_pred.append(true)
                    found_overlap = True
                    break

                # check for an overlap (not exact boundary match) with true entities
                elif pred.start_offset <= true.end_offset and true.start_offset <= pred.end_offset:
                    true_which_overlapped_with_pred.append(true)
                    if pred.e_type == true.e_type:  # overlaps with the same entity type
                        # overall results
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['correct'] += 1

                        # aggregated by entity type results
                        evaluation_agg_entities_type[pred.e_type]['strict']['incorrect'] += 1
                        evaluation_agg_entities_type[pred.e_type]['ent_type']['correct'] += 1

                        found_overlap = True
                        break

                    else:  # overlaps with a different entity type
                        # overall results
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['incorrect'] += 1

                        # aggregated by entity type results
                        evaluation_agg_entities_type[pred.e_type]['strict']['incorrect'] += 1
                        evaluation_agg_entities_type[pred.e_type]['ent_type']['incorrect'] += 1

                        found_overlap = True
                        break

            # count spurius (i.e., over-generated) entities
            if not found_overlap:
                # overall results
                evaluation['strict']['spurius'] += 1
                evaluation['ent_type']['spurius'] += 1

                # aggregated by entity type results
                evaluation_agg_entities_type[pred.e_type]['strict']['spurius'] += 1
                evaluation_agg_entities_type[pred.e_type]['ent_type']['spurius'] += 1

    # count missed entities
    for true in true_named_entities:
        if true in true_which_overlapped_with_pred:
            continue
        else:
            # overall results
            evaluation['strict']['missed'] += 1
            evaluation['ent_type']['missed'] += 1

            # for the agg. by e_type
            evaluation_agg_entities_type[true.e_type]['strict']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['ent_type']['missed'] += 1

    # Compute 'possible', 'actual', according to SemEval-2013 Task 9.1
    for eval_type in ['strict', 'ent_type']:
        correct = evaluation[eval_type]['correct']
        incorrect = evaluation[eval_type]['incorrect']
        partial = evaluation[eval_type]['partial']
        missed = evaluation[eval_type]['missed']
        spurius = evaluation[eval_type]['spurius']

        # possible: nr. annotations in the gold-standard which contribute to the final score
        evaluation[eval_type]['possible'] = correct + incorrect + partial + missed

        # actual: number of annotations produced by the NER system
        evaluation[eval_type]['actual'] = correct + incorrect + partial + spurius

        actual = evaluation[eval_type]['actual']
        possible = evaluation[eval_type]['possible']

        if eval_type == 'partial_matching':
            precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
            recall = (correct + 0.5 * partial) / possible if possible > 0 else 0
        else:
            precision = correct / actual if actual > 0 else 0
            recall = correct / possible if possible > 0 else 0

        evaluation[eval_type]['precision'] = precision
        evaluation[eval_type]['recall'] = recall

    return evaluation, evaluation_agg_entities_type


def metrics(y_true, y_pred, chunk_set):
    # y_true = convertv2s(y_true)
    # y_pred = convertv2s(y_pred)

    pred_token = []
    true_token = []
    for pred_labels in y_pred:
        pred_token.extend(pred_labels)
    for true_labels in y_true:
        true_token.extend(true_labels)

    # compute(get_chunks(true_token), get_chunks(pred_token))
    eval, _ = compute_metrics(collect_named_entities(true_token), collect_named_entities(pred_token), chunk_set)
    print("::F1-score::", 2 * eval['strict']['precision'] * eval['strict']['recall'] / (
                eval['strict']['precision'] + eval['strict']['recall']))
    print("::eval_metrics::")
    pprint(eval['strict'])

"""
[ based on SemEval 2013 task 9.1 evaluation metriscs ]

Metric code from
https://github.com/davidsbatista/NER-Evaluation
"""
from copy import deepcopy
from collections import namedtuple
from pprint import pprint

import utils_saveNload
import _metrics_F1


Entity = namedtuple("Entity", "e_type start_offset end_offset")


def collect_named_entities(tokens):
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.
    :param tokens: a list of labels
    :return: a list of Entity named-tuples
    """

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, token_tag in enumerate(tokens):

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

        elif ent_type != token_tag[2:] or (ent_type == token_tag[2:] and token_tag[:1] == 'B'):

            end_offset = offset - 1
            named_entities.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token
    if ent_type and start_offset and end_offset is None:
        named_entities.append(Entity(ent_type, start_offset, len(tokens) - 1))

    return named_entities


def compute_metrics(true_named_entities, pred_named_entities):
    incorrect_type = dict.fromkeys(chunk_list(), 0)
    eval_metrics = {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0,
                    'inc_type': deepcopy(incorrect_type)}
    # target_tags_no_schema = ['MISC', 'PER', 'LOC', 'ORG']
    # target_tags_no_schema = chunk_list()

    # overall results
    evaluation = {'strict': deepcopy(eval_metrics),
                  'ent_type': deepcopy(eval_metrics),
                  'partial': deepcopy(eval_metrics),
                  'exact': deepcopy(eval_metrics)}

    # results by entity type
    evaluation_agg_entities_type = {e: deepcopy(evaluation) for e in chunk_list()}

    true_which_overlapped_with_pred = []  # keep track of entities that overlapped

    # go through each predicted named-entity
    for pred in pred_named_entities:
        found_overlap = False

        # Check each of the potential scenarios in turn. See
        # http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
        # for scenario explanation.

        # Scenario I: Exact match between true and pred

        if pred in true_named_entities:
            true_which_overlapped_with_pred.append(pred)
            evaluation['strict']['correct'] += 1
            evaluation['ent_type']['correct'] += 1
            evaluation['exact']['correct'] += 1
            evaluation['partial']['correct'] += 1

            # for the agg. by e_type results
            evaluation_agg_entities_type[pred.e_type]['strict']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['ent_type']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['exact']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['partial']['correct'] += 1

        else:

            # check for overlaps with any of the true entities

            for true in true_named_entities:

                pred_range = range(pred.start_offset, pred.end_offset)
                true_range = range(true.start_offset, true.end_offset)

                # Scenario IV: Offsets match, but entity type is wrong

                if true.start_offset == pred.start_offset and pred.end_offset == true.end_offset \
                        and true.e_type != pred.e_type:

                    # overall results
                    evaluation['strict']['incorrect'] += 1
                    evaluation['ent_type']['incorrect'] += 1
                    evaluation['partial']['correct'] += 1
                    evaluation['exact']['correct'] += 1

                    # aggregated by entity type results
                    evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                    evaluation_agg_entities_type[true.e_type]['ent_type']['incorrect'] += 1
                    evaluation_agg_entities_type[true.e_type]['partial']['correct'] += 1
                    evaluation_agg_entities_type[true.e_type]['exact']['correct'] += 1

                    # count incorrect prediction type
                    evaluation_agg_entities_type[true.e_type]['strict']['inc_type'][pred.e_type] += 1
                    evaluation_agg_entities_type[true.e_type]['ent_type']['inc_type'][pred.e_type] += 1

                    true_which_overlapped_with_pred.append(true)
                    found_overlap = True
                    break

                # check for an overlap i.e. not exact boundary match, with true entities

                elif find_overlap(true_range, pred_range):

                    true_which_overlapped_with_pred.append(true)

                    # Scenario V: There is an overlap (but offsets do not match
                    # exactly), and the entity type is the same.
                    # 2.1 overlaps with the same entity type

                    if pred.e_type == true.e_type:

                        # overall results
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['correct'] += 1
                        evaluation['partial']['partial'] += 1
                        evaluation['exact']['incorrect'] += 1

                        # aggregated by entity type results
                        evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                        evaluation_agg_entities_type[true.e_type]['ent_type']['correct'] += 1
                        evaluation_agg_entities_type[true.e_type]['partial']['partial'] += 1
                        evaluation_agg_entities_type[true.e_type]['exact']['incorrect'] += 1

                        # count incorrect prediction type
                        evaluation_agg_entities_type[true.e_type]['strict']['inc_type'][pred.e_type] += 1
                        evaluation_agg_entities_type[true.e_type]['exact']['inc_type'][pred.e_type] += 1

                        found_overlap = True
                        break

                    # Scenario VI: Entities overlap, but the entity type is
                    # different.

                    else:
                        # overall results
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['incorrect'] += 1
                        evaluation['partial']['partial'] += 1
                        evaluation['exact']['incorrect'] += 1

                        # aggregated by entity type results
                        # Results against the true entity

                        evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                        evaluation_agg_entities_type[true.e_type]['partial']['partial'] += 1
                        evaluation_agg_entities_type[true.e_type]['ent_type']['incorrect'] += 1
                        evaluation_agg_entities_type[true.e_type]['exact']['incorrect'] += 1

                        # count incorrect prediction type
                        evaluation_agg_entities_type[true.e_type]['strict']['inc_type'][pred.e_type] += 1
                        evaluation_agg_entities_type[true.e_type]['ent_type']['inc_type'][pred.e_type] += 1
                        evaluation_agg_entities_type[true.e_type]['exact']['inc_type'][pred.e_type] += 1

                        # Results against the predicted entity

                        # evaluation_agg_entities_type[pred.e_type]['strict']['spurious'] += 1

                        found_overlap = True
                        break

            # Scenario II: Entities are spurious (i.e., over-generated).

            if not found_overlap:
                # overall results
                evaluation['strict']['spurious'] += 1
                evaluation['ent_type']['spurious'] += 1
                evaluation['partial']['spurious'] += 1
                evaluation['exact']['spurious'] += 1

                # aggregated by entity type results
                evaluation_agg_entities_type[pred.e_type]['strict']['spurious'] += 1
                evaluation_agg_entities_type[pred.e_type]['ent_type']['spurious'] += 1
                evaluation_agg_entities_type[pred.e_type]['partial']['spurious'] += 1
                evaluation_agg_entities_type[pred.e_type]['exact']['spurious'] += 1

    # Scenario III: Entity was missed entirely.

    for true in true_named_entities:
        if true in true_which_overlapped_with_pred:
            continue
        else:
            # overall results
            evaluation['strict']['missed'] += 1
            evaluation['ent_type']['missed'] += 1
            evaluation['partial']['missed'] += 1
            evaluation['exact']['missed'] += 1

            # for the agg. by e_type
            evaluation_agg_entities_type[true.e_type]['strict']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['ent_type']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['partial']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['exact']['missed'] += 1

    # Compute 'possible', 'actual' according to SemEval-2013 Task 9.1 on the
    # overall results, and use these to calculate precision and recall.

    for eval_type in evaluation:
        evaluation[eval_type] = compute_actual_possible(evaluation[eval_type])

    # Compute 'possible', 'actual', and precision and recall on entity level
    # results. Start by cycling through the accumulated results.

    for entity_type, entity_level in evaluation_agg_entities_type.items():

        # Cycle through the evaluation types for each dict containing entity
        # level results.

        for eval_type in entity_level:
            evaluation_agg_entities_type[entity_type][eval_type] = compute_actual_possible(
                entity_level[eval_type]
            )

    return evaluation, evaluation_agg_entities_type


def find_overlap(true_range, pred_range):
    """Find the overlap between two ranges
    Find the overlap between two ranges. Return the overlapping values if
    present, else return an empty set().
    Examples:
    >>> find_overlap((1, 2), (2, 3))
    2
    >>> find_overlap((1, 2), (3, 4))
    set()
    """

    true_set = set(true_range)
    pred_set = set(pred_range)

    overlaps = true_set.intersection(pred_set)

    return overlaps


def compute_actual_possible(results):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with actual, possible populated.
    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    correct = results['correct']
    incorrect = results['incorrect']
    partial = results['partial']
    missed = results['missed']
    spurious = results['spurious']

    # Possible: number annotations in the gold-standard which contribute to the
    # final score

    possible = correct + incorrect + partial + missed

    # Actual: number of annotations produced by the NER system

    actual = correct + incorrect + partial + spurious

    results["actual"] = actual
    results["possible"] = possible

    return results


def compute_precision_recall(results, partial_or_type=False):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with precison and recall populated.
    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    actual = results["actual"]
    possible = results["possible"]
    partial = results['partial']
    correct = results['correct']

    if partial_or_type:
        precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
        recall = (correct + 0.5 * partial) / possible if possible > 0 else 0

    else:
        precision = correct / actual if actual > 0 else 0
        recall = correct / possible if possible > 0 else 0

    results["precision"] = precision
    results["recall"] = recall

    return results


def compute_precision_recall_wrapper(results):
    """
    Wraps the compute_precision_recall function and runs on a dict of results
    """

    results_a = {key: compute_precision_recall(value, True) for key, value in results.items() if
                 key in ['partial', 'ent_type']}
    results_b = {key: compute_precision_recall(value) for key, value in results.items() if
                 key in ['strict', 'exact']}

    results = {**results_a, **results_b}

    return results


def metrics_new(test_sents_labels, y_pred):
    incorrect_type = dict.fromkeys(chunk_list(), 0)
    metrics_results = {'correct': 0, 'incorrect': 0, 'partial': 0,
                       'missed': 0, 'spurious': 0, 'possible': 0, 'actual': 0}
                       # 'inc_type': deepcopy(incorrect_type)}
    metrics_results_type = {'correct': 0, 'incorrect': 0, 'partial': 0,
                       'missed': 0, 'spurious': 0, 'possible': 0, 'actual': 0,
                       'inc_type': deepcopy(incorrect_type)}

    # overall results
    # results = {'strict': deepcopy(metrics_results),
    #            'exact': deepcopy(metrics_results),
    #            'ent_type': deepcopy(metrics_results)
    #             }
    # eval_schema: strict, exact, partial, ent_type
    results = {'partial': deepcopy(metrics_results)}
    results_type = {'partial': deepcopy(metrics_results_type)}

    # results aggregated by entity type
    evaluation_agg_entities_type = {e: deepcopy(results_type) for e in chunk_list()}

    for true_ents, pred_ents in zip(test_sents_labels, y_pred):
        # compute results for one message
        tmp_results, tmp_agg_results = compute_metrics(collect_named_entities(true_ents),
                                                       collect_named_entities(pred_ents))

        # aggregate overall results
        for eval_schema in results.keys():
            for metric in metrics_results.keys():
                # print("check: ", tmp_results[eval_schema][metric])
                results[eval_schema][metric] += tmp_results[eval_schema][metric]

        # aggregate results by entity type
        for e_type in chunk_list():
            for eval_schema in tmp_agg_results[e_type]:
                for metric in tmp_agg_results[e_type][eval_schema]:
                    # print(metric)
                    # print(tmp_agg_results[e_type][eval_schema][metric])
                    if metric == 'inc_type':
                        for inc_type in tmp_agg_results[e_type][eval_schema][metric]:
                            # print(inc_type)
                            try:
                                evaluation_agg_entities_type[e_type][eval_schema][metric][inc_type] += \
                                    tmp_agg_results[e_type][eval_schema][metric][inc_type]
                            except KeyError:
                                pass
                    else:
                        try:
                            evaluation_agg_entities_type[e_type][eval_schema][metric] += \
                                tmp_agg_results[e_type][eval_schema][metric]
                        except KeyError:
                            pass

    _metrics_F1.show_result(results)

    print(":: overall results ::")
    pprint(results)

    print("\n:: results by entity type ::")
    pprint(evaluation_agg_entities_type)


def chunk_list():
    chunk_set = utils_saveNload.load_chunk_set()
    BI_removed_chk_list = []
    for ch in chunk_set:
        BI_removed_chk_list.append(ch[2:])
    return BI_removed_chk_list


if __name__ == "__main__":
    y_true = utils_saveNload.load_pickle_chr_ture()
    y_pred = utils_saveNload.load_pickle_chr_pred()
    metrics_new(y_true, y_pred)

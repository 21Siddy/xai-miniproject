import json
from owlapy.model import IRI, OWLNamedIndividual
from ontolearn.knowledge_base import KnowledgeBase

def get_nodes_dict(hg):
    n_dict = {}
    for n in hg.ntypes:
        n_dict[n] = hg.num_nodes(n)
    return n_dict


def get_lp_mutag_fid(gnn_pred_dt_train, gnn_pred_dt_test, idx_map):
    # function tro create learning problems for the Mutag dataset for fidelity evaluations based on GNN model predictions.
    train_positive_examples = [
        idx_map[item]["IRI"]
        for item in gnn_pred_dt_train
        if gnn_pred_dt_train[item] == 1
    ]
    train_negative_examples = [
        idx_map[item]["IRI"]
        for item in gnn_pred_dt_train
        if gnn_pred_dt_train[item] == 0
    ]

    # Positive and negative examples for test set
    test_positive_examples = [
        idx_map[item]["IRI"] for item in gnn_pred_dt_test if gnn_pred_dt_test[item] == 1
    ]
    test_negative_examples = [
        idx_map[item]["IRI"] for item in gnn_pred_dt_test if gnn_pred_dt_test[item] == 0
    ]
    assert (
        len(set(train_positive_examples).intersection(set(train_negative_examples)))
        == 0
    )

    lp_dict_test_train = {
        "carcino": {
            "positive_examples_train": train_positive_examples,
            "negative_examples_train": train_negative_examples,
            "positive_examples_test": test_positive_examples,
            "negative_examples_test": test_negative_examples,
        }
    }
    return lp_dict_test_train


def get_lp_bgs_fid(gnn_pred_dt_train, gnn_pred_dt_test, idx_map):
    # function tro create learning problems for the Mutag dataset for fidelity evaluations based on GNN model predictions.
    train_positive_examples = [
        idx_map[item]["IRI"]
        for item in gnn_pred_dt_train
        if gnn_pred_dt_train[item] == 1
    ]
    train_negative_examples = [
        idx_map[item]["IRI"]
        for item in gnn_pred_dt_train
        if gnn_pred_dt_train[item] == 0
    ]

    # Positive and negative examples for test set
    test_positive_examples = [
        idx_map[item]["IRI"] for item in gnn_pred_dt_test if gnn_pred_dt_test[item] == 1
    ]
    test_negative_examples = [
        idx_map[item]["IRI"] for item in gnn_pred_dt_test if gnn_pred_dt_test[item] == 0
    ]
    assert (
        len(set(train_positive_examples).intersection(set(train_negative_examples)))
        == 0
    )

    lp_dict_test_train = {
        "lithogenesis": {
            "positive_examples_train": train_positive_examples,
            "negative_examples_train": train_negative_examples,
            "positive_examples_test": test_positive_examples,
            "negative_examples_test": test_negative_examples,
        }
    }
    return lp_dict_test_train


def get_lp_aifb_fid(gnn_pred_dt_train, gnn_pred_dt_test, idx_map):
    # function tro create learning problems for the Mutag dataset for fidelity evaluations based on GNN model predictions.
    train_positive_examples = [
        idx_map[item]["IRI"]
        for item in gnn_pred_dt_train
        if gnn_pred_dt_train[item] == 1
    ]
    train_negative_examples = [
        idx_map[item]["IRI"]
        for item in gnn_pred_dt_train
        if gnn_pred_dt_train[item] == 0
    ]

    # Positive and negative examples for test set
    test_positive_examples = [
        idx_map[item]["IRI"] for item in gnn_pred_dt_test if gnn_pred_dt_test[item] == 1
    ]
    test_negative_examples = [
        idx_map[item]["IRI"] for item in gnn_pred_dt_test if gnn_pred_dt_test[item] == 0
    ]
    assert (
        len(set(train_positive_examples).intersection(set(train_negative_examples)))
        == 0
    )

    lp_dict_test_train = {
        "id1instance": {
            "positive_examples_train": train_positive_examples,
            "negative_examples_train": train_negative_examples,
            "positive_examples_test": test_positive_examples,
            "negative_examples_test": test_negative_examples,
        }
    }
    return lp_dict_test_train


def calculate_metrics(true_labels, pred_labels):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, pred_labels)

    # Calculate precision, recall, f1-score, and support for binary class
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="binary"
    )

    # Create a dictionary with metrics
    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1-score": f1_score,
    }

    return metrics_dict


def gen_evaluations(prediction_performance, explanation_performance):

    performances_evauations = {
        "pred_accuracy": prediction_performance["accuracy"],
        "pred_precision": prediction_performance["precision"],
        "pred_recall": prediction_performance["recall"],
        "pred_f1_score": prediction_performance["f1-score"],
        "exp_accuracy": explanation_performance["accuracy"],
        "exp_precision": explanation_performance["precision"],
        "exp_recall": explanation_performance["recall"],
        "exp_f1_score": explanation_performance["f1-score"],
    }

    return performances_evauations

def get_lp_videogames_fid(train_truths, test_truths, idx_map):
    # Path to your OWL ontology for EvoLearner
    kg_path = "data/KGs/videogame_f.rdf"

    kg = KnowledgeBase(path=kg_path)

    learning_problems = {}

    # Define the target concept URI for EvoLearner.
    # This can be a placeholder URI; it's what EvoLearner will try to find a DL expression for.
    target_concept_uri = "http://example.org/videogame#MultiGenreConcept"

    pos_train_uris = []
    neg_train_uris = []
    pos_test_uris = []
    neg_test_uris = []

    # Populate train examples based on GNN's train_truths
    for didx, label in train_truths.items():
        entry = idx_map[didx]
        # if entry is {"IRI": "..."} unwrap it, else assume it's already a string
        uri = entry["IRI"] if isinstance(entry, dict) else entry
        if label == 1:
            pos_train_uris.append(uri)
        else:
            neg_train_uris.append(uri)

    # Populate test examples in the same way
    for didx, label in test_truths.items():
        entry = idx_map[didx]
        uri = entry["IRI"] if isinstance(entry, dict) else entry
        if label == 1:
            pos_test_uris.append(uri)
        else:
            neg_test_uris.append(uri)

    learning_problems[target_concept_uri] = {
        "positive_examples_train": [OWLNamedIndividual(IRI.create(uri)) for uri in pos_train_uris],
        "negative_examples_train": [OWLNamedIndividual(IRI.create(uri)) for uri in neg_train_uris],
        "positive_examples_test": [OWLNamedIndividual(IRI.create(uri)) for uri in pos_test_uris],
        "negative_examples_test": [OWLNamedIndividual(IRI.create(uri)) for uri in neg_test_uris],
        "concept_individuals": [], # Not directly used by EvoLearner input from here
    }

    print(f"Prepared learning problem for concept '{target_concept_uri}':")
    print(f"  Train Pos: {len(pos_train_uris)}, Neg: {len(neg_train_uris)}")
    print(f"  Test Pos: {len(pos_test_uris)}, Neg: {len(neg_test_uris)}")

    return learning_problems, kg

import json
import os
import time

from ontolearn.concept_learner import EvoLearner
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1, Accuracy
from owlapy.model import IRI, OWLNamedIndividual
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calcuate_metrics(predictions_data):
    for _, examples in predictions_data.items():
        concept_individuals = set(examples["concept_individuals"])
        positive_examples = set(examples["positive_examples"])
        negative_examples = set(examples["negative_examples"])

        all_examples = positive_examples.union(negative_examples)

        true_labels = [1 if item in positive_examples else 0 for item in all_examples]
        pred_labels = [1 if item in concept_individuals else 0 for item in all_examples]

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


def train_evo(learning_problems, kg=None):

    if learning_problems is None:
        print("Learning problems is None. Exiting.")
        return {}, 0, {}

    if kg is None:
        print("Knowledge Graph (kg) is None. Exiting.")
        return {}, 0, {}

    if isinstance(kg, str):
        if kg == "aifb":
            path_kb = "KGs/aifb-hetero.owl"
        elif kg == "mutag":
            path_kb = "KGs/mutag-hetero.owl"
        elif kg == "bgs":
            path_kb = "KGs/bgs-hetero.owl"
        elif kg == "videogames":
            path_kb = "KGs/videogame_f.rdf"
        else:
            raise ValueError(f"Unknown KG: {kg}")
        target_kb = KnowledgeBase(path=path_kb)
    else:
        target_kb = kg

    t0 = time.time()
    explanation_dict = {}
    predictions_dict = {}

    for str_target_concept, examples in learning_problems.items():
        positive_examples_train = set(examples["positive_examples_train"])
        negative_examples_train = set(examples["negative_examples_train"])

        # Validate that individuals exist in the knowledge base
        for ind in positive_examples_train:
            if ind not in target_kb._ind_set:
                print(f"WARNING: Positive example individual {ind.get_iri().as_str()} not found in KnowledgeBase._ind_set. This might lead to 'NoneType' errors later.")
        for ind in negative_examples_train:
            if ind not in target_kb._ind_set:
                print(f"WARNING: Negative example individual {ind.get_iri().as_str()} not found in KnowledgeBase._ind_set. This might lead to 'NoneType' errors later.")


        typed_pos = set(positive_examples_train)
        typed_neg = set(negative_examples_train)

        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

        model = EvoLearner(
            knowledge_base=target_kb, max_runtime=600, quality_func=Accuracy()
        )
        
        try:
            model.fit(lp, verbose=False)
        except AttributeError as e:
            if "'NoneType' object has no attribute 'namespace'" in str(e):
                print(f"\nCaught AttributeError: {e}")
                print(f"This indicates an individual could not be resolved by Ontolearn/Owlready2 internally.")
                print(f"Learning Problem Positive Examples (IRIs): {[ind.get_iri().as_str() for ind in lp.pos]}")
                print(f"Learning Problem Negative Examples (IRIs): {[ind.get_iri().as_str() for ind in lp.neg]}")
                print(f"Please check if these individuals are correctly defined and referenced in the knowledge base.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during model.fit: {e}")
            raise


        # Get Top n hypotheses
        hypotheses = list(model.best_hypotheses(n=3))
        if not hypotheses:
            print("EvoLearner did not find any hypotheses. Skipping prediction for this target.")
            continue
            
        [print(_) for _ in hypotheses]
        
        positive_examples_test = set(examples["positive_examples_test"])
        negative_examples_test = set(examples["negative_examples_test"])
        all_test_examples = positive_examples_test.union(negative_examples_test)

        best_concept = hypotheses[0].concept
        # Get all individuals that belong to the learned concept as a set of string IRIs
        individuals_in_learned_concept = {
            indv.get_iri().as_str()
            for indv in target_kb.individuals_set(best_concept)
        }
        
        concept_length = target_kb.concept_len(hypotheses[0].concept)

        # Create the prediction dictionary with STRING IRIs as keys
        for individual in all_test_examples:
            individual_iri_str = individual.get_iri().as_str()
            # Predict 1 if the test individual is in the set of individuals for the learned concept
            if individual_iri_str in individuals_in_learned_concept:
                predictions_dict[individual_iri_str] = 1
            else:
                predictions_dict[individual_iri_str] = 0

        explanation_dict[str_target_concept] = {
            "best_concept": str(best_concept),
            "concept_length": concept_length,
        }

    t1 = time.time()
    duration = t1 - t0
    return predictions_dict, duration, explanation_dict

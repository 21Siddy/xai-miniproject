import json
import os
from statistics import mean
from prettytable import PrettyTable

def print_results(model_name="RGCN"):
    """
    Scans the results directory to dynamically discover all datasets and explainers,
    and prints a formatted table of the evaluation metrics.
    """
    results_dir = "results/evaluations"
    table = PrettyTable()
    table.field_names = [
        "Model", "Dataset", "Pred Accuracy", "Pred Precision", 
        "Pred Recall", "Pred F1 Score", "Exp Accuracy", 
        "Exp Precision", "Exp Recall", "Exp F1 Score"
    ]

    all_results = []
    
    # Dynamically find all explainer directories for the given model
    model_dir = os.path.join(results_dir, model_name)
    if not os.path.exists(model_dir):
        print(f"No results found for model: {model_name}")
        return

    for explainer_name in sorted(os.listdir(model_dir)):
        explainer_dir = os.path.join(model_dir, explainer_name)
        if os.path.isdir(explainer_dir):
            # Find all result files (e.g., aifb.json, videogames.json)
            for result_file in sorted(os.listdir(explainer_dir)):
                if result_file.endswith(".json"):
                    dataset_name = result_file.replace(".json", "")
                    
                    try:
                        with open(os.path.join(explainer_dir, result_file), 'r') as f:
                            data = json.load(f)
                            
                        # Handle case where file might be empty
                        if not data:
                            continue

                        # Use the metrics from the first run in the JSON file
                        # This assumes at least one run is present
                        metrics = list(data.values())[0]
                        row = [
                            explainer_name, dataset_name,
                            round(metrics.get("pred_accuracy", 0), 3),
                            round(metrics.get("pred_precision", 0), 3),
                            round(metrics.get("pred_recall", 0), 3),
                            round(metrics.get("pred_f1_score", 0), 3),
                            round(metrics.get("exp_accuracy", 0), 3),
                            round(metrics.get("exp_precision", 0), 3),
                            round(metrics.get("exp_recall", 0), 3),
                            round(metrics.get("exp_f1_score", 0), 3)
                        ]
                        all_results.append(row)
                    except (IOError, json.JSONDecodeError, IndexError) as e:
                        print(f"Warning: Could not read or parse {result_file}. Error: {e}")
                        continue
                        
    # Sort the collected results by dataset, then by explainer model
    all_results.sort(key=lambda x: (x[1], x[0])) 
    
    # Add all collected rows to the table
    for row in all_results:
        table.add_row(row)

    print(table)

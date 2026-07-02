import mlflow

def compare_and_find_best_run():
    print("--- Querying MLflow Backend for Best Run --- \n")
    
    # 1. Set the active target experiment
    experiment_name = "Heart_Disease_Classification"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Error: Experiment '{experiment_name}' not found.")
        return

    # 2. Programmatically query all runs under this experiment identifier
    # Sorts descending based on accuracy so the top performer is element 0
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"]
    )
    
    # Display the full experiment summary table for the instructor
    print("All Tracked Runs Ordered by Accuracy:")
    columns_to_show = [
        "run_id", 
        "params.n_estimators", 
        "params.max_depth", 
        "metrics.accuracy", 
        "metrics.f1_score"
    ]
    # Filter to only show columns if they exist in the dataframe
    existing_cols = [col for col in columns_to_show if col in runs_df.columns]
    print(runs_df[existing_cols].to_string(index=False))
    print("\n" + "="*50)
    
    # 3. Extract top performer properties
    best_run = runs_df.iloc[0]
    
    print("🌟 Programmatic Winner Found! 🌟")
    print(f"Run ID:      {best_run['run_id']}")
    print(f"Accuracy:    {best_run['metrics.accuracy']:.4f}")
    print(f"F1-Score:    {best_run['metrics.f1_score']:.4f}")
    print(f"Estimators:  {best_run['params.n_estimators']}")
    print(f"Max Depth:   {best_run['params.max_depth']}")
    print(f"Data Hash:   {best_run['params.dvc_data_version_hash']}")
    print("="*50)

if __name__ == "__main__":
    compare_and_find_best_run()
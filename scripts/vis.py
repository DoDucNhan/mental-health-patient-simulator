import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy import stats

def analyze_evaluation_results():
    # Load automatic evaluation results
    auto_results = {}
    for filename in os.listdir():
        if filename.startswith("evaluation_results_") and filename.endswith(".json"):
            with open(filename, 'r') as f:
                results = json.load(f)
                auto_results.update(results)
    
    # Load expert evaluation results
    expert_files = [f for f in os.listdir() if f.startswith("expert_evaluation_") and f.endswith(".json")]
    
    expert_data = []
    for file in expert_files:
        with open(file, 'r') as f:
            data = json.load(f)
            for key, evaluation in data.items():
                expert_data.append(evaluation)
    
    # Convert to DataFrame
    expert_df = pd.DataFrame(expert_data)
    
    # Load trainee evaluation results
    trainee_files = [f for f in os.listdir() if f.startswith("trainee_evaluation_") and f.endswith(".json")]
    
    trainee_data = []
    for file in trainee_files:
        with open(file, 'r') as f:
            data = json.load(f)
            trainee_data.append(data)
    
    trainee_df = pd.DataFrame(trainee_data)
    
    # Analysis 1: Compare models based on automatic evaluation
    auto_comparison = {}
    for model_name, scores in auto_results.items():
        auto_comparison[model_name] = scores["combined"]
    
    auto_df = pd.DataFrame(auto_comparison)
    
    # Create radar chart for automatic evaluation
    plt.figure(figsize=(12, 8))
    
    # Compute number of variables
    categories = list(auto_df.index)
    N = len(categories)
    
    # Create angle for each variable
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize the radar plot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Set y limits
    ax.set_ylim(0, 10)
    
    # Plot each model
    for model_name in auto_df.columns:
        values = auto_df[model_name].values.flatten().tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Automatic Evaluation Results", size=20)
    plt.savefig("automatic_evaluation_radar.png", dpi=300, bbox_inches='tight')
    
    # Analysis 2: Expert evaluations by model
    # Aggregate expert evaluations by actual model
    expert_agg = expert_df.groupby("actual_model").agg({
        "active_listening": "mean",
        "empathy": "mean",
        "maladaptive_cognitions": "mean",
        "emotional_states": "mean",
        "conversational_style": "mean",
        "overall_fidelity": "mean",
        "open_mindedness": "mean"
    }).reset_index()
    
    # Create bar chart for expert evaluations
    plt.figure(figsize=(14, 8))
    expert_agg_melted = pd.melt(expert_agg, id_vars=["actual_model"], 
                                var_name="metric", value_name="score")
    
    sns.barplot(x="metric", y="score", hue="actual_model", data=expert_agg_melted)
    plt.title("Expert Evaluation Scores by Model", size=16)
    plt.xlabel("Evaluation Metric", size=14)
    plt.ylabel("Average Score (1-10)", size=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig("expert_evaluation_bar.png", dpi=300)
    
    # Analysis 3: Trainee preferences and confidence
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="selected_model", y="confidence", data=trainee_df)
    plt.title("Trainee Confidence by Selected Model", size=16)
    plt.xlabel("Model", size=14)
    plt.ylabel("Confidence (1-10)", size=14)
    plt.tight_layout()
    plt.savefig("trainee_confidence_box.png", dpi=300)
    
    plt.figure(figsize=(12, 6))
    trainee_metrics = trainee_df[["selected_model", "realism", "helpfulness"]]
    trainee_metrics_melted = pd.melt(trainee_metrics, id_vars=["selected_model"], 
                                    var_name="metric", value_name="score")
    
    sns.barplot(x="selected_model", y="score", hue="metric", data=trainee_metrics_melted)
    plt.title("Trainee Evaluation by Model", size=16)
    plt.xlabel("Model", size=14)
    plt.ylabel("Average Score (1-10)", size=14)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig("trainee_evaluation_bar.png", dpi=300)
    
    # Statistical testing: ANOVA between models
    metrics = ["active_listening", "empathy", "maladaptive_cognitions", 
               "emotional_states", "conversational_style", "overall_fidelity", 
               "open_mindedness"]
    
    anova_results = {}
    for metric in metrics:
        groups = []
        for model in expert_df["actual_model"].unique():
            group = expert_df[expert_df["actual_model"] == model][metric].values
            groups.append(group)
        
        f_val, p_val = stats.f_oneway(*groups)
        anova_results[metric] = {
            "f_value": f_val,
            "p_value": p_val,
            "significant": p_val < 0.05
        }
    
    # Create comparison table
    summary_table = {
        "Model": [],
        "Auto Eval Avg": [],
        "Expert Eval Avg": [],
        "Trainee Confidence": [],
        "Trainee Realism": [],
        "Trainee Helpfulness": []
    }
    
    for model in auto_df.columns:
        auto_avg = auto_df[model].mean()
        
        # Expert average if available
        if model in expert_agg["actual_model"].values:
            expert_row = expert_agg[expert_agg["actual_model"] == model]
            expert_avg = expert_row.iloc[:, 1:].mean(axis=1).values[0]
        else:
            expert_avg = np.nan
        
        # Trainee metrics if available
        trainee_model_data = trainee_df[trainee_df["selected_model"] == model]
        if not trainee_model_data.empty:
            trainee_confidence = trainee_model_data["confidence"].mean()
            trainee_realism = trainee_model_data["realism"].mean()
            trainee_helpfulness = trainee_model_data["helpfulness"].mean()
        else:
            trainee_confidence = np.nan
            trainee_realism = np.nan
            trainee_helpfulness = np.nan
        
        summary_table["Model"].append(model)
        summary_table["Auto Eval Avg"].append(auto_avg)
        summary_table["Expert Eval Avg"].append(expert_avg)
        summary_table["Trainee Confidence"].append(trainee_confidence)
        summary_table["Trainee Realism"].append(trainee_realism)
        summary_table["Trainee Helpfulness"].append(trainee_helpfulness)
    
    summary_df = pd.DataFrame(summary_table)
    
    # Save results
    summary_df.to_csv("evaluation_summary.csv", index=False)
    
    with open("anova_results.json", "w") as f:
        json.dump(anova_results, f, indent=2)
    
    # Return summary data
    return {
        "auto_results": auto_df,
        "expert_results": expert_agg,
        "trainee_results": trainee_df,
        "summary": summary_df,
        "anova_results": anova_results
    }

if __name__ == "__main__":
    results = analyze_evaluation_results()
    
    # Print summary
    print("Evaluation Summary:")
    print(results["summary"])
    
    print("\nANOVA Results:")
    for metric, stats in results["anova_results"].items():
        significance = "Significant" if stats["significant"] else "Not significant"
        print(f"{metric}: p-value = {stats['p_value']:.4f} ({significance})")
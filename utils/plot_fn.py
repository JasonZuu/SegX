import numpy as np
import matplotlib.pyplot as plt

from utils.metrics import calculate_p_value


def few_shot_plot_fn(test_results: dict, model_names: list, performance_metric: str, plt_save_path: str = None):
    """
    Updated plotting function to display prediction performance (AUROC or AUPRC)
    and fairness (EO) metrics in a single plot with bar and line charts, filtering for ratios >= 0.25,
    changing x-axis to number of samples, and ensuring alignment of points across models at each x-coordinate.
    
    :param test_results: A dictionary containing test results for different models.
                         Each key is a model identifier, and the values are dictionaries
                         with metrics at different ratios of training data used.
    :param model_names: List of model identifiers to be plotted.
    :param performance_metric: The performance metric to plot ('auroc' or 'auprc').
    :param plt_save_path: The path to save the plot. If None, the plot is displayed instead.
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC']  # Assign a unique color to each model
    ax2 = ax1.twinx()
    
    num_models = len(model_names)
    # Filtering ratios and converting them into number of samples
    filtered_ratios = [ratio for ratio in next(iter(test_results.values())) if float(ratio) >= 0.25]
    num_ratios = len(filtered_ratios)
    total_width = 0.8
    bar_width = total_width / num_models
    index = np.arange(num_ratios)  # x locations for the groups
    
    for i, model in enumerate(model_names):
        ratios = []
        performance_means = []
        eo_means = []
        performance_cis = []
        
        sorted_ratios = sorted([ratio for ratio in test_results[model] if float(ratio) >= 0.25], key=lambda x: float(x))
        for ratio in sorted_ratios:
            metrics = test_results[model][ratio]
            ratios.append(ratio)
            performance_means.append(metrics[performance_metric]['mean'])
            performance_cis.append(metrics[performance_metric]['ci'])
            eo_means.append(max(metrics['eo[tp]']['mean'], metrics['eo[fp]']['mean']))
        
        position_ax1 = index + (i - num_models / 2) * bar_width + bar_width / 2
        position_ax2 = index
        ax1.bar(position_ax1, performance_means, bar_width, color=colors[i], alpha=0.35, label=f'{model} {performance_metric.upper()}',
                yerr=performance_cis, ecolor=colors[i])
        ax2.plot(position_ax2, eo_means, color=colors[i], linestyle="-", marker="o", label=f'{model} EO')
    
    ax1.set_xlabel('Number of Samples Used for Training')
    ax1.set_ylabel(f'Performance Metric ({performance_metric.upper()})')
    ax2.set_ylabel('Fairness Metric (EO)')
    
    if performance_metric == 'auroc':
        ax1.set_ylim(0.6, 1.0)
    elif performance_metric == 'auprc':
        ax1.set_ylim(0.4, 1.0)
    
    ax2.set_ylim(0.0, 0.21)
    
    # Adjusting x-ticks to reflect number of samples instead of ratio
    ax1.set_xticks(index)
    ax1.set_xticklabels([str(int(float(ratio) * 6000)) for ratio in sorted_ratios])    
    ax1.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    
    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    
    plt.title('Prediction Performance and Fairness for Data-Scarce Simulation')
    plt.tight_layout()
    
    if plt_save_path is not None:
        plt.savefig(plt_save_path)
    else:
        plt.show()


def few_shot_plot_fn_old(*test_results, model_names, metric:str, plt_save_path: str=None, show_ci: bool=True):
    """
    plotting function for few-shot simulation results

    :param test_results: A dictionary containing test results for different models.
                         Each key is a model identifier, and the value is another
                         dictionary with 'roauc' and 'equalized_odds' metrics.
    :param plt_save_path: The path to save the plot to. If None, the plot will be
                        displayed instead of saved.
    :param show_ci: Whether to show the confidence intervals of the plotted values.
    """
    # Plot set up
    plt.figure(figsize=(8, 6))
    
    # Plotting the roauc values
    for model, test_result in zip(model_names, test_results):  # each test_result is for a model
        num_samples_list = []
        metric_mean_list = []
        metric_ci_list = []
        for used_ratio, results in test_result.items():
            num_samples = float(used_ratio) * 6000
            metric_mean = results[metric]['mean']
            metric_ci = results[metric]['ci']

            num_samples_list.append(num_samples)
            metric_mean_list.append(metric_mean)
            metric_ci_list.append(metric_ci)

        if show_ci:
            plt.errorbar(num_samples_list, metric_mean_list, yerr=metric_ci_list, fmt='-o', label=model)
        else:
            plt.plot(num_samples_list, metric_mean_list, '-o', label=model)

    plt.title(f'{metric.upper()} Performance')
    plt.xlabel('Number of Training Images')
    plt.ylabel(metric.upper())
    plt.legend()

    plt.tight_layout()
    if plt_save_path is not None:
        plt.savefig(plt_save_path)
    else:
        plt.show()


def draw_curve_plot(results):
    # Extracting keys and values from the input data
    keys = sorted(results.keys())
    roauc_means = [results[key]['roauc']['mean'] for key in keys]
    roauc_stds = [results[key]['roauc']['std'] for key in keys]
    equalized_odds_means = [results[key]['equalized_odds']['mean'] for key in keys]
    equalized_odds_stds = [results[key]['equalized_odds']['std'] for key in keys]

    # Plotting ROAUC
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.errorbar(keys, roauc_means, yerr=roauc_stds, fmt='-o', capsize=5, label='ROAUC')
    plt.xlabel('Mask Used Ratio')
    plt.ylabel('ROAUC')
    plt.title('ROAUC vs Mask Used Ratio')
    plt.ylim(0.85, 1.0)

    # Plotting Equalized Odds
    plt.subplot(1, 2, 2)
    plt.errorbar(keys, equalized_odds_means, yerr=equalized_odds_stds, fmt='-o', capsize=5, label='Equalized Odds')
    plt.xlabel('Mask Used Ratio')
    plt.ylabel('Equalized Odds')
    plt.title('Equalized Odds vs Mask Used Ratio')
    plt.ylim(0.0, 0.15)

    plt.tight_layout()
    plt.show()


def draw_column_with_ci(result_dict: dict):
    # Data
    models = list(result_dict.keys())
    roauc_means = [model_result['roauc']['mean'] for model_result in result_dict.values()]
    roauc_cis = [model_result['roauc']['ci'] for model_result in result_dict.values()]
    n_trials = 5

    # Confidence intervals (CI) calculation: CI = mean ± z * ci / sqrt(n)
    # For 95% confidence level, z-score is approximately 1.96
    z_score = 1.96
    ci_upper = [mean + z_score * ci / np.sqrt(n_trials) for mean, ci in zip(roauc_means, roauc_cis)]
    ci_lower = [mean - z_score * ci / np.sqrt(n_trials) for mean, ci in zip(roauc_means, roauc_cis)]

    # Plotting
    plt.figure(figsize=(10, 6))
    bar_positions = np.arange(len(models))
    plt.bar(bar_positions, roauc_means, yerr=[np.subtract(roauc_means, ci_lower), np.subtract(ci_upper, roauc_means)],
            capsize=5, alpha=0.7, color='yellowgreen', edgecolor='black')
    plt.ylim(0.75, 1)
    plt.xticks(bar_positions, models, rotation=45)
    plt.ylabel('ROAUC')
    plt.title('ROAUC Scores with Confidence Intervals (CI) for Different Models')
    plt.grid(axis='y')

    plt.show()


def calculate_p_value_from_dict(result_dict):
    baseline_val = result_dict['baseline']
    for model_name, val in result_dict.items():
        if model_name == 'baseline':
            continue
        for metric_name, metric in val.items():
            if metric_name == 'f1':
                continue
            p_value = calculate_p_value(metric['mean'], metric['std'], 5, baseline_val[metric_name]['mean'], baseline_val[metric_name]['std'], 5)
            print(f"{model_name} {metric_name} {p_value}")

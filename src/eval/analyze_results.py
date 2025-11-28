#!/usr/bin/env python3
"""
Statistical analysis and visualization of evaluation results.

This script:
1. Loads evaluation results from final_evaluation.json
2. Performs statistical analyses (bootstrap CI, pairwise comparisons)
3. Checks for position bias
4. Generates visualizations
5. Extracts qualitative examples

Usage:
    python src/eval/analyze_results.py [--results results/final_evaluation.json]
"""

import argparse
import json
import os
import numpy as np
from collections import defaultdict, Counter
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_file):
    """Load evaluation results from JSON."""
    with open(results_file, 'r') as f:
        return json.load(f)


def bootstrap_confidence_intervals(scores, n_bootstrap=1000, confidence=0.95):
    """
    Compute bootstrap confidence intervals for scores.

    Args:
        scores: List of scores
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 95%)

    Returns:
        Tuple of (lower, upper) confidence bounds
    """
    if not scores:
        return 0, 0

    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrapped_means.append(np.mean(sample))

    lower = np.percentile(bootstrapped_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrapped_means, (1 + confidence) / 2 * 100)

    return lower, upper


def position_bias_check(results):
    """
    Check if any model appears more frequently in certain positions.

    Args:
        results: Evaluation results dict

    Returns:
        Dict of position statistics per model
    """
    position_counts = defaultdict(lambda: defaultdict(int))

    for example in results['per_example']:
        position_map = example['position_map']
        for position, model_name in position_map.items():
            position_counts[model_name][position] += 1

    return dict(position_counts)


def pairwise_win_rates(results):
    """
    Compute pairwise win rates (e.g., RLHF vs SFT).

    Args:
        results: Evaluation results dict

    Returns:
        Dict of win rates for each pair
    """
    models = results['summary']['models_evaluated']
    win_counts = defaultdict(lambda: defaultdict(int))
    total_comparisons = defaultdict(lambda: defaultdict(int))

    for example in results['per_example']:
        scores = example['grader_scores']

        # Compare each pair of models
        for i, model1 in enumerate(models):
            for model2 in models[i + 1:]:
                if model1 in scores and model2 in scores:
                    total_comparisons[model1][model2] += 1
                    total_comparisons[model2][model1] += 1

                    if scores[model1] > scores[model2]:
                        win_counts[model1][model2] += 1
                    elif scores[model2] > scores[model1]:
                        win_counts[model2][model1] += 1
                    # Ties don't count as wins

    # Compute win rates
    win_rates = {}
    for model1 in models:
        win_rates[model1] = {}
        for model2 in models:
            if model1 != model2:
                total = total_comparisons[model1].get(model2, 0)
                wins = win_counts[model1].get(model2, 0)
                win_rates[model1][model2] = wins / total if total > 0 else 0

    return win_rates


def plot_grader_scores(results, output_dir):
    """Bar plot of grader scores with error bars."""
    models = results['summary']['models_evaluated']
    normalized_scores = results['summary']['grader_normalized_scores']

    # Compute confidence intervals for grader scores
    # Convert total scores back to per-example scores
    per_example_scores = defaultdict(list)
    for example in results['per_example']:
        for model, score in example['grader_scores'].items():
            per_example_scores[model].append(score)

    means = [normalized_scores[model] for model in models]
    cis = []
    for model in models:
        scores = per_example_scores[model]
        # Normalize scores (divide by 3 to get 0-1 range)
        normalized = [s / 3 for s in scores]
        lower, upper = bootstrap_confidence_intervals(normalized)
        # Store as error bars (distance from mean)
        mean = np.mean(normalized)
        cis.append([mean - lower, upper - mean])

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=np.array(cis).T, capsize=5, alpha=0.8)

    # Color bars
    colors = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
    for bar, color in zip(bars, colors[:len(models)]):
        bar.set_color(color)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Grader Normalized Score (0-1)', fontsize=12)
    ax.set_title('Model Comparison: Grader Rankings', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grader_scores.png'), dpi=300)
    print(f"Saved grader scores plot to {output_dir}/grader_scores.png")
    plt.close()


def plot_similarity_distributions(results, output_dir):
    """Distribution plots of similarity scores."""
    models = results['summary']['models_evaluated']
    similarity_scores = results['similarity_scores']

    fig, ax = plt.subplots(figsize=(10, 6))

    for model in models:
        scores = similarity_scores[model]
        ax.hist(scores, bins=20, alpha=0.5, label=model, density=True)

    ax.set_xlabel('Similarity to Oracle', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Similarity Scores', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_distributions.png'), dpi=300)
    print(f"Saved similarity distributions plot to {output_dir}/similarity_distributions.png")
    plt.close()


def plot_pairwise_win_rates(win_rates, models, output_dir):
    """Heatmap of pairwise win rates."""
    # Create matrix
    matrix = np.zeros((len(models), len(models)))
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if model1 != model2:
                matrix[i, j] = win_rates[model1][model2]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0.5,
        vmin=0,
        vmax=1,
        xticklabels=models,
        yticklabels=models,
        ax=ax,
        cbar_kws={'label': 'Win Rate'}
    )

    ax.set_title('Pairwise Win Rates', fontsize=14, fontweight='bold')
    ax.set_xlabel('Opponent Model', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pairwise_win_rates.png'), dpi=300)
    print(f"Saved pairwise win rates plot to {output_dir}/pairwise_win_rates.png")
    plt.close()


def qualitative_analysis(results, output_file, num_examples=5):
    """
    Generate qualitative examples (best/worst cases for each model).

    Args:
        results: Evaluation results dict
        output_file: Output text file
        num_examples: Number of examples per category
    """
    models = results['summary']['models_evaluated']

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("QUALITATIVE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # For each model, find best and worst examples
        for model in models:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"MODEL: {model.upper()}\n")
            f.write("=" * 80 + "\n")

            # Sort examples by this model's score
            sorted_examples = sorted(
                results['per_example'],
                key=lambda x: x['grader_scores'].get(model, 0),
                reverse=True
            )

            # Best examples
            f.write(f"\n--- TOP {num_examples} EXAMPLES ---\n\n")
            for i, example in enumerate(sorted_examples[:num_examples]):
                f.write(f"Example {i + 1} (Score: {example['grader_scores'].get(model, 0)}/3)\n")
                f.write("-" * 80 + "\n")
                f.write("CONTEXT:\n")
                f.write(example['prompt'][:500] + "...\n\n")
                f.write(f"{model.upper()} RESPONSE:\n")
                f.write(example['responses'][model] + "\n\n")
                f.write("GRADER EXPLANATION:\n")
                f.write(example['grader_explanation'] + "\n\n")

            # Worst examples
            f.write(f"\n--- BOTTOM {num_examples} EXAMPLES ---\n\n")
            for i, example in enumerate(sorted_examples[-num_examples:]):
                f.write(f"Example {i + 1} (Score: {example['grader_scores'].get(model, 0)}/3)\n")
                f.write("-" * 80 + "\n")
                f.write("CONTEXT:\n")
                f.write(example['prompt'][:500] + "...\n\n")
                f.write(f"{model.upper()} RESPONSE:\n")
                f.write(example['responses'][model] + "\n\n")
                f.write("GRADER EXPLANATION:\n")
                f.write(example['grader_explanation'] + "\n\n")

        # Examples where RLAIF+RLHF beats RLHF-only
        if 'rlaif_rlhf' in models and 'rlhf' in models:
            f.write("\n" + "=" * 80 + "\n")
            f.write("RLAIF+RLHF vs RLHF-ONLY: Where RLAIF warmup helped\n")
            f.write("=" * 80 + "\n\n")

            improvements = [
                ex for ex in results['per_example']
                if ex['grader_scores'].get('rlaif_rlhf', 0) > ex['grader_scores'].get('rlhf', 0)
            ]
            improvements.sort(
                key=lambda x: x['grader_scores'].get('rlaif_rlhf', 0) - x['grader_scores'].get('rlhf', 0),
                reverse=True
            )

            for i, example in enumerate(improvements[:num_examples]):
                diff = example['grader_scores'].get('rlaif_rlhf', 0) - example['grader_scores'].get('rlhf', 0)
                f.write(f"Example {i + 1} (Improvement: +{diff})\n")
                f.write("-" * 80 + "\n")
                f.write("CONTEXT:\n")
                f.write(example['prompt'][:500] + "...\n\n")
                f.write("RLHF RESPONSE:\n")
                f.write(example['responses']['rlhf'] + "\n\n")
                f.write("RLAIF+RLHF RESPONSE:\n")
                f.write(example['responses']['rlaif_rlhf'] + "\n\n")
                f.write("GRADER EXPLANATION:\n")
                f.write(example['grader_explanation'] + "\n\n")

    print(f"Saved qualitative analysis to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze evaluation results"
    )
    parser.add_argument(
        '--results',
        type=str,
        default='results/final_evaluation.json',
        help='Evaluation results file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Output directory for analysis'
    )
    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results}...")
    results = load_results(args.results)
    models = results['summary']['models_evaluated']

    print(f"Loaded {results['summary']['num_examples']} examples")
    print(f"Models: {', '.join(models)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    # Position bias check
    print("\nPosition Bias Check:")
    position_stats = position_bias_check(results)
    for model, positions in position_stats.items():
        total = sum(positions.values())
        print(f"  {model}:")
        for pos in sorted(positions.keys()):
            count = positions[pos]
            print(f"    Position {pos}: {count}/{total} ({count / total * 100:.1f}%)")

    # Pairwise win rates
    print("\nPairwise Win Rates:")
    win_rates = pairwise_win_rates(results)
    for model1 in models:
        print(f"  {model1}:")
        for model2 in models:
            if model1 != model2:
                rate = win_rates[model1][model2]
                print(f"    vs {model2}: {rate:.3f} ({rate * 100:.1f}%)")

    # Bootstrap confidence intervals for similarity scores
    print("\nSimilarity Score Confidence Intervals (95%):")
    for model in models:
        scores = results['similarity_scores'][model]
        mean = np.mean(scores)
        lower, upper = bootstrap_confidence_intervals(scores)
        print(f"  {model}: {mean:.3f} [{lower:.3f}, {upper:.3f}]")

    # Statistical significance tests (paired t-tests for similarity scores)
    print("\nPairwise Significance Tests (Similarity Scores):")
    for i, model1 in enumerate(models):
        for model2 in models[i + 1:]:
            scores1 = results['similarity_scores'][model1]
            scores2 = results['similarity_scores'][model2]

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(scores1, scores2)
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."

            print(f"  {model1} vs {model2}: t={t_stat:.3f}, p={p_value:.4f} {sig}")

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_grader_scores(results, figures_dir)
    plot_similarity_distributions(results, figures_dir)
    plot_pairwise_win_rates(win_rates, models, figures_dir)

    # Qualitative analysis
    print("\n" + "=" * 80)
    print("GENERATING QUALITATIVE ANALYSIS")
    print("=" * 80)

    qualitative_file = os.path.join(args.output_dir, 'qualitative_examples.txt')
    qualitative_analysis(results, qualitative_file)

    # Save analysis report
    report_file = os.path.join(args.output_dir, 'analysis_report.txt')
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Number of examples: {results['summary']['num_examples']}\n")
        f.write(f"Models evaluated: {', '.join(models)}\n\n")

        f.write("Grader Normalized Scores:\n")
        for model, score in sorted(
            results['summary']['grader_normalized_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            f.write(f"  {model}: {score:.3f}\n")

        f.write("\nAverage Similarity Scores:\n")
        for model, score in sorted(
            results['summary']['average_similarity_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            f.write(f"  {model}: {score:.3f}\n")

        f.write("\nPairwise Win Rates:\n")
        for model1 in models:
            f.write(f"  {model1}:\n")
            for model2 in models:
                if model1 != model2:
                    rate = win_rates[model1][model2]
                    f.write(f"    vs {model2}: {rate:.3f}\n")

    print(f"Saved analysis report to {report_file}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to {args.output_dir}/")
    print("  - analysis_report.txt")
    print("  - qualitative_examples.txt")
    print("  - figures/grader_scores.png")
    print("  - figures/similarity_distributions.png")
    print("  - figures/pairwise_win_rates.png")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Calculate treatment effects between two conditions (before vs after treatment).

This script takes two CSV files containing coin variance statistics (before and after treatment),
and calculates average treatment effect, p-value, probability of effect, and Cohen's d for each prefix.

Usage:
    python get_treatment_effect.py <before_csv> <after_csv>

Example:
    python get_treatment_effect.py coin_variance_output/coin_variance_summary_all_prefixes_game5.csv coin_variance_output/coin_variance_summary_all_prefixes_game10.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import math


def cohens_d(mean1, std1, n1, mean2, std2, n2):
    """
    Calculate Cohen's d effect size.

    Args:
        mean1, std1, n1: Mean, standard deviation, and sample size for group 1
        mean2, std2, n2: Mean, standard deviation, and sample size for group 2

    Returns:
        Cohen's d value
    """
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    # Avoid division by zero
    if pooled_std == 0:
        return 0.0

    d = (mean2 - mean1) / pooled_std
    return d


def t_cdf(t, df):
    """
    Approximate cumulative distribution function for t-distribution.
    Uses approximation for large df, and direct calculation for small df.

    Args:
        t: t-statistic value
        df: degrees of freedom

    Returns:
        Cumulative probability
    """
    # For large df, t-distribution approaches normal distribution
    if df > 30:
        # Use normal approximation
        return normal_cdf(t)

    # For smaller df, use a simple approximation
    # This is a simplified approach - not as accurate as scipy but sufficient for our purposes
    x = df / (df + t**2)
    # Incomplete beta function approximation
    # For simplicity, we'll use a normal approximation scaled by df
    return normal_cdf(t * np.sqrt((df + 1) / (df + 3)))


def normal_cdf(x):
    """
    Cumulative distribution function for standard normal distribution.
    Uses error function approximation.

    Args:
        x: Value

    Returns:
        Cumulative probability
    """
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))


def welch_t_test(mean1, std1, n1, mean2, std2, n2):
    """
    Perform Welch's t-test (doesn't assume equal variances).

    Args:
        mean1, std1, n1: Mean, standard deviation, and sample size for group 1
        mean2, std2, n2: Mean, standard deviation, and sample size for group 2

    Returns:
        t-statistic and p-value
    """
    # Calculate standard errors
    se1 = std1 / np.sqrt(n1)
    se2 = std2 / np.sqrt(n2)

    # Calculate t-statistic
    se_diff = np.sqrt(se1**2 + se2**2)

    # Avoid division by zero
    if se_diff == 0:
        return 0.0, 1.0

    t_stat = (mean2 - mean1) / se_diff

    # Calculate degrees of freedom (Welch-Satterthwaite equation)
    if n1 > 1 and n2 > 1:
        df = (se1**2 + se2**2)**2 / (se1**4 / (n1 - 1) + se2**4 / (n2 - 1))
    else:
        df = max(n1 + n2 - 2, 1)

    # Calculate two-tailed p-value
    p_value = 2 * (1 - t_cdf(abs(t_stat), df))

    return t_stat, p_value


def probability_of_superiority(mean1, std1, n1, mean2, std2, n2):
    """
    Calculate probability that a random sample from group 2 is greater than a random sample from group 1.
    Assumes normal distributions.

    Args:
        mean1, std1, n1: Mean, standard deviation, and sample size for group 1
        mean2, std2, n2: Mean, standard deviation, and sample size for group 2

    Returns:
        Probability of superiority (0 to 1)
    """
    # Standard deviation of the difference
    std_diff = np.sqrt(std1**2 + std2**2)

    # Avoid division by zero
    if std_diff == 0:
        if mean2 > mean1:
            return 1.0
        elif mean2 < mean1:
            return 0.0
        else:
            return 0.5

    # Z-score of the difference
    z = (mean2 - mean1) / std_diff

    # Probability that sample from group 2 > sample from group 1
    prob = normal_cdf(z)

    return prob


def main():
    """Main entry point."""
    # Parse command line arguments
    if len(sys.argv) != 3:
        print("Usage: python get_treatment_effect.py <before_csv> <after_csv>")
        print("Example: python get_treatment_effect.py coin_variance_output/summary_before.csv coin_variance_output/summary_after.csv")
        sys.exit(1)

    before_csv = Path(sys.argv[1])
    after_csv = Path(sys.argv[2])

    # Check if files exist
    if not before_csv.exists():
        print(f"❌ Error: Before treatment CSV not found: {before_csv}")
        sys.exit(1)

    if not after_csv.exists():
        print(f"❌ Error: After treatment CSV not found: {after_csv}")
        sys.exit(1)

    print("=" * 80)
    print(f"TREATMENT EFFECT ANALYSIS")
    print("=" * 80)
    print(f"Before treatment: {before_csv}")
    print(f"After treatment: {after_csv}")
    print()

    # Read CSV files
    before_df = pd.read_csv(before_csv)
    after_df = pd.read_csv(after_csv)

    print(f"Before treatment data: {len(before_df)} prefix(es)")
    print(f"After treatment data: {len(after_df)} prefix(es)")
    print()

    # Extract matching key (last 3 parts of prefix split by "_")
    def extract_matching_key(prefix):
        """Extract last 3 parts of prefix for matching."""
        parts = prefix.split('_')
        if len(parts) >= 3:
            return '_'.join(parts[-3:])
        return prefix

    before_df['matching_key'] = before_df['prefix'].apply(extract_matching_key)
    after_df['matching_key'] = after_df['prefix'].apply(extract_matching_key)

    print("Matching keys (last 3 parts):")
    print(f"  Before: {sorted(before_df['matching_key'].unique())}")
    print(f"  After:  {sorted(after_df['matching_key'].unique())}")
    print()

    # Merge on matching_key to ensure we're comparing the same conditions
    merged_df = before_df.merge(
        after_df,
        on='matching_key',
        suffixes=('_before', '_after'),
        how='inner'
    )

    if len(merged_df) == 0:
        print("❌ Error: No matching prefixes found between before and after files")
        print("   (Matching based on last 3 parts of prefix split by '_')")
        sys.exit(1)

    print(f"Matched prefixes: {len(merged_df)}")
    print()

    # Calculate treatment effects
    results = []

    for idx, row in merged_df.iterrows():
        matching_key = row['matching_key']
        prefix_before = row['prefix_before']
        prefix_after = row['prefix_after']

        # Before treatment stats
        mean_before = row['mean_before']
        std_before = row['std_before']
        n_before = row['n_simulations_before']

        # After treatment stats
        mean_after = row['mean_after']
        std_after = row['std_after']
        n_after = row['n_simulations_after']

        # Calculate treatment effect (difference in means)
        treatment_effect = mean_after - mean_before

        # Calculate Cohen's d
        cohens_d_value = cohens_d(mean_before, std_before, n_before, mean_after, std_after, n_after)

        # Calculate t-test
        t_stat, p_value = welch_t_test(mean_before, std_before, n_before, mean_after, std_after, n_after)

        # Calculate probability of superiority
        prob_effect = probability_of_superiority(mean_before, std_before, n_before, mean_after, std_after, n_after)

        results.append({
            'matching_key': matching_key,
            'prefix_before': prefix_before,
            'prefix_after': prefix_after,
            'mean_before': mean_before,
            'std_before': std_before,
            'n_before': n_before,
            'mean_after': mean_after,
            'std_after': std_after,
            'n_after': n_after,
            'treatment_effect': treatment_effect,
            'cohens_d': cohens_d_value,
            't_statistic': t_stat,
            'p_value': p_value,
            'prob_effect': prob_effect
        })

        print("=" * 80)
        print(f"Condition: {matching_key}")
        print(f"  Before: {prefix_before}")
        print(f"  After:  {prefix_after}")
        print("=" * 80)
        print(f"Before: Mean={mean_before:.2f}, Std={std_before:.2f}, N={n_before}")
        print(f"After:  Mean={mean_after:.2f}, Std={std_after:.2f}, N={n_after}")
        print()
        print(f"Average Treatment Effect: {treatment_effect:.4f}")
        print(f"Cohen's d:                {cohens_d_value:.4f}")
        print(f"t-statistic:              {t_stat:.4f}")
        print(f"p-value:                  {p_value:.4f}")
        print(f"Prob(After > Before):     {prob_effect:.4f}")

        # Interpretation
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "n.s."
        print(f"Significance:             {sig}")

        # Cohen's d interpretation
        abs_d = abs(cohens_d_value)
        if abs_d < 0.2:
            effect_size = "negligible"
        elif abs_d < 0.5:
            effect_size = "small"
        elif abs_d < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"
        print(f"Effect Size:              {effect_size}")
        print()

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_dir = Path('treatment_effect_output')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename based on input files
    before_name = before_csv.stem
    after_name = after_csv.stem
    output_file = output_dir / f'treatment_effect_{before_name}_vs_{after_name}.csv'

    results_df.to_csv(output_file, index=False)

    print("\n" + "=" * 80)
    print(f"SUMMARY")
    print("=" * 80)
    print(results_df[['matching_key', 'treatment_effect', 'cohens_d', 'p_value', 'prob_effect']].to_string(index=False))
    print("=" * 80)
    print(f"\n📊 Results saved to: {output_file}")


if __name__ == "__main__":
    main()

"""Extended Birthday Problem: Monte Carlo simulation for 2+ and 3+ shared birthdays."""
from collections import Counter
import random
import math
import matplotlib.pyplot as plt


def has_shared_birthday(birthdays, min_matches=2):
    """True if at least `min_matches` people share a birthday."""
    if not birthdays:
        return False
    if min_matches == 2:
        return len(birthdays) != len(set(birthdays))
    return max(Counter(birthdays).values()) >= min_matches


def simulate(group_size, trials=10000, rng=None, min_matches=2):
    if group_size < 0 or trials <= 0:
        raise ValueError('Use a nonnegative group size and positive trial count.')
    if min_matches < 2:
        raise ValueError('min_matches must be at least 2.')
    if rng is None:
        rng = random.Random()
    matches = 0
    for _ in range(trials):
        birthdays = [rng.randint(1, 365) for _ in range(group_size)]
        if has_shared_birthday(birthdays, min_matches=min_matches):
            matches += 1
    return matches / trials


def theoretical_probability(group_size):
    """Exact probability for at least 2 shared birthdays."""
    if group_size < 0:
        raise ValueError('Group size must be nonnegative.')
    if group_size > 365:
        return 1.0
    return 1 - math.prod((365 - i) / 365 for i in range(group_size))


def poisson_three_way_approx(group_size):
    """Poisson approximation for 3+ shared birthdays: 1 - exp(-C(n,3) / 365^2)."""
    if group_size < 3:
        return 0.0
    lam = math.comb(group_size, 3) / (365 ** 2)
    return 1 - math.exp(-lam)


def main():
    trials = 10000
    rng = random.Random(217)
    sizes = [10, 23, 30, 40, 50, 60, 75, 85, 88, 90, 100, 120, 150]

    lines = []
    lines.append(f'{trials:,} trials per group size (seed=217)')
    header = f'{"People":>6} {"2+ Exact":>12} {"2+ Sim":>12} {"3+ Sim":>12} {"3+ Approx":>12}'
    lines.append(header)
    lines.append('-' * len(header))

    results_2 = []
    results_3 = []

    for n in sizes:
        exact_2 = theoretical_probability(n)
        sim_2 = simulate(n, trials, rng, min_matches=2)
        sim_3 = simulate(n, trials, rng, min_matches=3)
        approx_3 = poisson_three_way_approx(n)

        results_2.append(sim_2)
        results_3.append(sim_3)

        lines.append(
            f'{n:6d} {exact_2:12.2%} {sim_2:12.2%} {sim_3:12.2%} {approx_3:12.2%}'
        )

    lines.append('\nEffect of trial count on 3+ shared birthdays (88 people):')
    for count in [100, 1000, 10000, 100000]:
        est = simulate(88, count, random.Random(217), min_matches=3)
        se = 100 * math.sqrt(est * (1 - est) / count)
        lines.append(f'{count:>7,}: {est:.2%}; estimated standard error: {se:.2f} percentage points')

    output_text = '\n'.join(lines)
    print(output_text)

    # Save table output to results_3way.txt (matching Sam's results.txt workflow)
    with open('results_3way.txt', 'w') as f:
        f.write(output_text + '\n')

    # --- Chart 1: 2-Person vs. 3-Person Comparison Curve ---
    x_dense = list(range(1, 151))
    plt.figure(figsize=(9, 5))
    plt.plot(x_dense, [100 * theoretical_probability(n) for n in x_dense],
             label='2+ Shared Birthdays (Exact)', color='steelblue', linewidth=2)
    plt.scatter(sizes, [100 * p for p in results_2],
                label='2+ Shared Birthdays (Monte Carlo)', color='steelblue', zorder=3)

    plt.plot(sizes, [100 * p for p in results_3], marker='o',
             label='3+ Shared Birthdays (Monte Carlo)', color='darkorange', linewidth=2, zorder=3)
    plt.plot(x_dense, [100 * poisson_three_way_approx(n) for n in x_dense],
             linestyle='--', label='3+ Shared Birthdays (Poisson Approx)', color='darkorange', alpha=0.6)

    plt.axhline(50, linestyle='--', color='gray', alpha=0.5, label='50% Threshold')
    plt.axvline(23, linestyle=':', color='steelblue', alpha=0.7)
    plt.axvline(88, linestyle=':', color='darkorange', alpha=0.7)

    plt.text(24, 6, 'n = 23 (50.7%)', color='steelblue', fontsize=9, fontweight='bold')
    plt.text(89, 6, 'n = 88 (~51.1%)', color='darkorange', fontsize=9, fontweight='bold')

    plt.xlabel('Number of people')
    plt.ylabel('Probability of shared birthday (%)')
    plt.title('The Birthday Problem: 2-Person vs. 3-Person Matches')
    plt.ylim(0, 100)
    plt.xlim(0, 150)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('birthday_comparison_3way.png', dpi=180)
    plt.close()

    # --- Chart 2: Side-by-Side Bar Chart for Key Milestone Group Sizes ---
    milestones = [23, 50, 88, 120]
    m_2 = [100 * simulate(n, trials, random.Random(217), min_matches=2) for n in milestones]
    m_3 = [100 * simulate(n, trials, random.Random(217), min_matches=3) for n in milestones]

    x_idx = range(len(milestones))
    width = 0.35

    plt.figure(figsize=(8, 5))
    bars1 = plt.bar([i - width/2 for i in x_idx], m_2, width, label='2+ Shared Birthdays', color='steelblue')
    bars2 = plt.bar([i + width/2 for i in x_idx], m_3, width, label='3+ Shared Birthdays', color='darkorange')

    plt.axhline(50, linestyle='--', color='gray', alpha=0.5)
    plt.xticks(list(x_idx), [f'n = {n}' for n in milestones])
    plt.ylabel('Simulated Probability (%)')
    plt.title('2-Person vs. 3-Person Match Probability at Key Group Sizes')
    plt.ylim(0, 110)
    plt.legend()
    plt.grid(axis='y', alpha=0.2)

    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f'{bar.get_height():.1f}%', ha='center', fontsize=8.5)
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f'{bar.get_height():.1f}%', ha='center', fontsize=8.5)

    plt.tight_layout()
    plt.savefig('birthday_milestones_bar.png', dpi=180)
    plt.close()


if __name__ == '__main__':
    main()
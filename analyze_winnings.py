#!/usr/bin/env python3
"""
Script to run poker engine, parse game results, and plot winnings distribution.

This script:
1. Runs engine.py using the .venv Python interpreter
2. Parses gamelog.txt to extract player A's final winnings
3. Plots the distribution of A's winnings using matplotlib
"""
import subprocess
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def run_engine(venv_python_path: str, engine_path: str) -> None:
    """
    Run the poker engine using the specified Python interpreter.
    
    Args:
        venv_python_path: Path to the Python interpreter in .venv
        engine_path: Path to engine.py
        
    Raises:
        subprocess.CalledProcessError: If engine execution fails
    """
    print("Running poker engine...")
    result = subprocess.run(
        [venv_python_path, engine_path],
        capture_output=True,
        text=True,
        check=False
    )
    
    if result.returncode != 0:
        print(f"Engine stderr:\n{result.stderr}", file=sys.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, 
            [venv_python_path, engine_path],
            result.stdout,
            result.stderr
        )
    
    print("Engine completed successfully")
    if result.stdout:
        print(f"Engine output:\n{result.stdout}")


def parse_gamelog(gamelog_path: str) -> list[int]:
    """
    Parse gamelog.txt to extract player A's final winnings from each game.
    
    The format we're looking for is:
        Final, A (XXX), B (YYY)
    
    where XXX is A's total winnings (positive or negative).
    
    Args:
        gamelog_path: Path to gamelog.txt
        
    Returns:
        List of A's winnings for each game
        
    Raises:
        FileNotFoundError: If gamelog doesn't exist
        ValueError: If no valid game results found
    """
    print(f"Parsing gamelog: {gamelog_path}")
    
    if not Path(gamelog_path).exists():
        raise FileNotFoundError(f"Gamelog not found: {gamelog_path}")
    
    # Pattern to match: Final, A (255), B (-255)
    # Captures the number in parentheses after A
    pattern = r'^Final,\s*A\s*\((-?\d+)\),\s*B\s*\((-?\d+)\)'
    
    winnings = []
    
    with open(gamelog_path, 'r') as f:
        for line in f:
            match = re.match(pattern, line.strip())
            if match:
                a_winnings = int(match.group(1))
                b_winnings = int(match.group(2))
                
                # Sanity check: A's winnings should equal negative of B's winnings
                assert a_winnings == -b_winnings, (
                    f"Inconsistent winnings: A={a_winnings}, B={b_winnings}. "
                    f"Expected A = -B"
                )
                
                winnings.append(a_winnings)
    
    if not winnings:
        raise ValueError(f"No game results found in {gamelog_path}")
    
    print(f"Found {len(winnings)} game results")
    return winnings[0]


def plot_winnings_distribution(winnings: list[int], output_path: str = None) -> None:
    """
    Plot the distribution of player A's winnings.
    
    Args:
        winnings: List of A's winnings for each game
        output_path: Optional path to save the plot (if None, displays interactively)
    """
    winnings_array = np.array(winnings)
    
    # Calculate statistics
    mean_winnings = np.mean(winnings_array)
    median_winnings = np.median(winnings_array)
    std_winnings = np.std(winnings_array)
    total_winnings = np.sum(winnings_array)
    
    print("\n=== Winnings Statistics ===")
    print(f"Total games: {len(winnings)}")
    print(f"Total winnings: {total_winnings}")
    print(f"Mean winnings per game: {mean_winnings:.2f}")
    print(f"Median winnings per game: {median_winnings:.2f}")
    print(f"Std deviation: {std_winnings:.2f}")
    print(f"Min winnings: {np.min(winnings_array)}")
    print(f"Max winnings: {np.max(winnings_array)}")
    print(f"Win rate: {100 * np.sum(winnings_array > 0) / len(winnings):.1f}%")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(winnings_array, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(mean_winnings, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_winnings:.1f}')
    ax1.axvline(median_winnings, color='green', linestyle='--', linewidth=2, label=f'Median: {median_winnings:.1f}')
    ax1.set_xlabel('Winnings per Game', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Distribution of Player A Winnings\n({len(winnings)} games)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative winnings over time
    cumulative_winnings = np.cumsum(winnings_array)
    ax2.plot(cumulative_winnings, linewidth=1.5)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Game Number', fontsize=12)
    ax2.set_ylabel('Cumulative Winnings', fontsize=12)
    ax2.set_title(f'Cumulative Winnings Over Time\n(Final: {total_winnings})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        print("\nDisplaying plot...")
        plt.show()


def main():
    """Main execution function."""
    # Setup paths
    workspace_root = Path(__file__).parent
    venv_python = workspace_root / ".venv" / "bin" / "python"
    engine_path = workspace_root / "engine.py"
    gamelog_path = workspace_root / "gamelog.txt"
    output_plot_path = workspace_root / "winnings_distribution.png"
    
    # Validate paths
    if not venv_python.exists():
        raise FileNotFoundError(
            f"Virtual environment Python not found at: {venv_python}\n"
            f"Expected .venv/bin/python in workspace root"
        )
    
    if not engine_path.exists():
        raise FileNotFoundError(f"Engine not found at: {engine_path}")
    
    try:
        winnings = []
        for i in range(100):
            # Step 1: Run the engine
            run_engine(str(venv_python), str(engine_path))
            
            # Step 2: Parse the gamelog
            winnings.append(parse_gamelog(str(gamelog_path)))
        
        # Step 3: Plot the distribution
        plot_winnings_distribution(winnings, output_path=str(output_plot_path))
        
        print("\n✓ Analysis complete!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


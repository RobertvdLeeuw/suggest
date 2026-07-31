import marimo

__generated_with = "0.14.11"
app = marimo.App(width="full")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm

    # Simulated data: results for each architecture across two metrics
    # Format: [Metric1, Metric2]
    results = np.array([
        [0.8, 0.6],
        [1.1, 1.0],
        [0.9, 0.8],
        [1.5, 1.2],
        [1.3, 1.0]
    ])

    # Standard deviations for confidence interval calculation
    std_devs = np.array([
        [0.05, 0.04],
        [0.1, 0.08],
        [0.07, 0.06],
        [0.15, 0.12],
        [0.12, 0.1]
    ])

    # Architectures and colors
    architectures = ['Arch1', 'Arch2', 'Arch3', 'Arch4', 'Arch5']
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    # Calculate the 95% confidence intervals
    z_score = norm.ppf(0.975)  # 95% confidence
    n_runs = 5  # Number of runs

    # Prepare the plot
    plt.figure(figsize=(10, 8))

    # Scatter plot with confidence intervals
    for i in range(len(results)):
        # Average results
        x, y = results[i]
    
        # Calculate confidence interval radius
        radius_x = z_score * (std_devs[i][0] / np.sqrt(n_runs))
        radius_y = z_score * (std_devs[i][1] / np.sqrt(n_runs))

        # Create shaded circle for confidence interval
        circle = plt.Circle((x, y), np.mean([radius_x, radius_y]), color=colors[i], alpha=0.2, label=architectures[i])
        plt.gca().add_artist(circle)

        # Scatter point
        plt.scatter(x, y, color=colors[i], edgecolor='black', s=100, zorder=5)

    # Customize the plot
    plt.title('Architectures Comparison with 95% Confidence Intervals')
    plt.xlabel('Metric 1 Results')
    plt.ylabel('Metric 2 Results')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.xlim(0, 2)  # Set limits based on your data range
    plt.ylim(0, 2)
    plt.tight_layout()

    # Display the plot
    plt.show()

    return


if __name__ == "__main__":
    app.run()

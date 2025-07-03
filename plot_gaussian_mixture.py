"""
Probability Density Function Plotting for Gaussian Mixtures
Author: Assistant
Date: July 2, 2025

This module provides utilities to plot the probability density functions of Gaussian mixtures.
It supports plotting individual components, the combined mixture, and comparison between different mixtures.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import List, Tuple, Optional
import matplotlib.colors as mcolors

def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Compute the probability density function of a Gaussian distribution.
    
    Args:
        x: Input values
        mu: Mean of the Gaussian
        sigma: Standard deviation of the Gaussian
    
    Returns:
        PDF values at input points
    """
    return norm.pdf(x, loc=mu, scale=sigma)

def gaussian_mixture_pdf(x: np.ndarray, weights: np.ndarray, means: np.ndarray, 
                        variances: np.ndarray) -> np.ndarray:
    """
    Compute the probability density function of a Gaussian mixture.
    
    Args:
        x: Input values
        weights: Mixture weights (should sum to 1)
        means: Means of each Gaussian component
        variances: Variances of each Gaussian component
    
    Returns:
        PDF values of the mixture at input points
    """
    # Ensure weights sum to 1
    weights = np.array(weights) / np.sum(weights)
    means = np.array(means)
    variances = np.array(variances)
    stds = np.sqrt(variances)
    
    pdf = np.zeros_like(x)
    for w, mu, sigma in zip(weights, means, stds):
        pdf += w * gaussian_pdf(x, mu, sigma)
    
    return pdf

def plot_gaussian_mixture(weights: np.ndarray, means: np.ndarray, variances: np.ndarray,
                         x_range: Optional[Tuple[float, float]] = None,
                         n_points: int = 1000,
                         show_components: bool = True,
                         show_mixture: bool = True,
                         title: str = "Gaussian Mixture PDF",
                         figsize: Tuple[int, int] = (10, 6),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the probability density function of a Gaussian mixture.
    
    Args:
        weights: Mixture weights
        means: Means of each component
        variances: Variances of each component
        x_range: Range for x-axis (min, max). If None, automatically determined
        n_points: Number of points for plotting
        show_components: Whether to show individual components
        show_mixture: Whether to show the combined mixture
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (optional)
    
    Returns:
        matplotlib Figure object
    """
    weights = np.array(weights)
    means = np.array(means)
    variances = np.array(variances)
    stds = np.sqrt(variances)
    
    # Ensure weights sum to 1
    weights = weights / np.sum(weights)
    
    # Determine x range if not provided
    if x_range is None:
        margin = 3  # 3 standard deviations
        x_min = np.min(means - margin * stds)
        x_max = np.max(means + margin * stds)
        x_range = (x_min, x_max)
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot individual components if requested
    if show_components:
        colors = plt.cm.Set1(np.linspace(0, 1, len(weights)))
        for i, (w, mu, sigma, color) in enumerate(zip(weights, means, stds, colors)):
            component_pdf = w * gaussian_pdf(x, mu, sigma)
            ax.plot(x, component_pdf, '--', color=color, alpha=0.7, 
                   label=f'Component {i+1}: w={w:.3f}, μ={mu:.2f}, σ={sigma:.2f}')
    
    # Plot combined mixture if requested
    if show_mixture:
        mixture_pdf = gaussian_mixture_pdf(x, weights, means, variances)
        ax.plot(x, mixture_pdf, 'k-', linewidth=2, label='Mixture PDF')
    
    # Formatting
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add text box with mixture parameters
    info_text = f'Components: {len(weights)}\n'
    info_text += f'Weights: {[f"{w:.3f}" for w in weights]}\n'
    info_text += f'Means: {[f"{mu:.2f}" for mu in means]}\n'
    info_text += f'Std devs: {[f"{std:.2f}" for std in stds]}'
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig

def compare_gaussian_mixtures(mixtures: List[dict], 
                             labels: Optional[List[str]] = None,
                             x_range: Optional[Tuple[float, float]] = None,
                             n_points: int = 1000,
                             title: str = "Gaussian Mixture Comparison",
                             figsize: Tuple[int, int] = (12, 6),
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare multiple Gaussian mixtures by plotting their PDFs.
    
    Args:
        mixtures: List of dictionaries, each containing 'weights', 'means', 'variances'
        labels: Labels for each mixture
        x_range: Range for x-axis (min, max). If None, automatically determined
        n_points: Number of points for plotting
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (optional)
    
    Returns:
        matplotlib Figure object
    """
    if labels is None:
        labels = [f'Mixture {i+1}' for i in range(len(mixtures))]
    
    # Determine x range if not provided
    if x_range is None:
        all_means = []
        all_stds = []
        for mixture in mixtures:
            all_means.extend(mixture['means'])
            all_stds.extend(np.sqrt(mixture['variances']))
        
        margin = 3
        x_min = min(all_means) - margin * max(all_stds)
        x_max = max(all_means) + margin * max(all_stds)
        x_range = (x_min, x_max)
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each mixture
    colors = plt.cm.tab10(np.linspace(0, 1, len(mixtures)))
    for i, (mixture, label, color) in enumerate(zip(mixtures, labels, colors)):
        pdf = gaussian_mixture_pdf(x, mixture['weights'], mixture['means'], mixture['variances'])
        ax.plot(x, pdf, linewidth=2, color=color, label=label)
    
    # Formatting
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig

def plot_with_samples(weights: np.ndarray, means: np.ndarray, variances: np.ndarray,
                     samples: Optional[np.ndarray] = None,
                     n_samples: int = 1000,
                     x_range: Optional[Tuple[float, float]] = None,
                     n_points: int = 1000,
                     title: str = "Gaussian Mixture PDF with Samples",
                     figsize: Tuple[int, int] = (10, 6),
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Gaussian mixture PDF with histogram of samples for verification.
    
    Args:
        weights: Mixture weights
        means: Means of each component
        variances: Variances of each component
        samples: Pre-generated samples (optional)
        n_samples: Number of samples to generate if samples not provided
        x_range: Range for x-axis
        n_points: Number of points for PDF plotting
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    weights = np.array(weights)
    means = np.array(means)
    variances = np.array(variances)
    
    # Generate samples if not provided
    if samples is None:
        samples = sample_gaussian_mixture(weights, means, variances, n_samples)
    
    # Determine x range
    if x_range is None:
        sample_range = (np.min(samples), np.max(samples))
        pdf_range_margin = 3 * np.sqrt(np.max(variances))
        x_min = min(sample_range[0], np.min(means) - pdf_range_margin)
        x_max = max(sample_range[1], np.max(means) + pdf_range_margin)
        x_range = (x_min, x_max)
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram of samples
    ax.hist(samples, bins=50, density=True, alpha=0.7, color='lightblue', 
            edgecolor='black', label='Sample Histogram')
    
    # Plot theoretical PDF
    pdf = gaussian_mixture_pdf(x, weights, means, variances)
    ax.plot(x, pdf, 'r-', linewidth=2, label='Theoretical PDF')
    
    # Formatting
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig

def sample_gaussian_mixture(weights: np.ndarray, means: np.ndarray, 
                           variances: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Generate samples from a Gaussian mixture model.
    
    Args:
        weights: Mixture weights
        means: Means of each component
        variances: Variances of each component
        n_samples: Number of samples to generate
    
    Returns:
        Array of samples
    """
    weights = np.array(weights) / np.sum(weights)
    means = np.array(means)
    stds = np.sqrt(variances)
    
    # Choose components based on weights
    component_indices = np.random.choice(len(weights), size=n_samples, p=weights)
    
    # Generate samples
    samples = np.zeros(n_samples)
    for i in range(len(weights)):
        mask = component_indices == i
        n_component_samples = np.sum(mask)
        if n_component_samples > 0:
            samples[mask] = np.random.normal(means[i], stds[i], n_component_samples)
    
    return samples

# Example usage and demonstrations
if __name__ == "__main__":
    # Example 1: Simple bimodal mixture
    print("Example 1: Bimodal Gaussian Mixture")
    # weights1 = [0.3, 0.7]
    # means1 = [-2, 3]
    # variances1 = [1, 0.5]
    means1 = [0.0138813,  0.57830552]
    variances1= [0.0028598 , 0.13632926]
    weights1= [0.96105375, 0.03894625]
    
    fig1 = plot_gaussian_mixture(weights1, means1, variances1, 
                                 title="Bimodal Gaussian Mixture",
                                 save_path="figures/bimodal_mixture.png")
    plt.show()

#%%
# visuzlize the the pdf of X given Y = 10
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal
import math

def get_distribution(mean: float, std: float) -> torch.distributions.Normal:
    """
    Create a Normal distribution with the given mean and standard deviation.

    Parameters
    ----------
    mean : float
        Mean of the Normal distribution.
    std : float
        Standard deviation of the Normal distribution.

    Returns
    -------
    torch.distributions.Normal
        A Normal distribution object.
    """
    return torch.distributions.Normal(mean, std)


def get_pdf(dist: torch.distributions.Normal, x: torch.Tensor) -> torch.Tensor: 
    """
    Compute the probability density function (PDF) of the distribution at the given x values.

    Parameters
    ----------
    dist : torch.distributions.Normal
        A Normal distribution object.
    x : torch.Tensor
        A tensor of x values.

    Returns
    -------
    torch.Tensor
        The PDF values at the given x values.
    """
    return dist.log_prob(x).exp()  # log_prob → use .exp() to get PDF


def get_distribution_viz(dist: Normal, x: torch.Tensor) -> None:
    """
    Visualize the distribution by plotting its PDF and samples.

    Parameters
    ----------
    dist : Normal
        A Normal distribution object.
    x : torch.Tensor
        A tensor of x values.
    """
    pdf = get_pdf(dist, x)
    samples = dist.sample((1000,))  # Draw 1000 samples
    # Plot histogram of samples
    plt.hist(samples.numpy(), bins=50, density=True, alpha=0.6, label='Sampled')
    plt.plot(x.numpy(), pdf.numpy(), label=f'N({dist.loc}, {dist.scale**2})')
    plt.title('Normal Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.legend()
    plt.show()


# %%
if __name__ == '__main__':
    mean = 2 + math.sqrt(1.25) * (10 - 2)
    std = 0.5

    dist = get_distribution(mean, std)
    x = torch.linspace(5, 15, 1000)
    get_distribution_viz(dist, x)
# %%
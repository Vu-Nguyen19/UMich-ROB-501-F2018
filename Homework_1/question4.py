#%%
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal

#%%
def get_distribution(mean: float, std: float) -> Normal:
    """
    Create a Normal distribution with the given mean and standard deviation.

    Parameters
    ----------
    mean : float
        Mean of the distribution.
    std : float
        Standard deviation of the distribution.

    Returns
    -------
    Normal
        A Normal distribution object.
    """
    return Normal(loc=mean, scale=std)

def get_pdf(dist: Normal, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the probability density function (PDF) of the distribution at the given x values.

    Parameters
    ----------
    dist : Normal
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

#%%
if __name__ == "__main__":
    # question 4a) Create 2 Normal distributions with different means and standard deviations
    # and plot their PDFs
    # Create two Normal distributions
    mean = 0
    std1 = 1
    dist1 = get_distribution(mean, std1)
    std2 = 3
    dist2 = get_distribution(mean, std2)

    # Create a range of x values
    x = torch.linspace(-20, 20, 1000)  # from -100 to 100 with 1000 points

    # Compute the probability density function (PDF)
    pdf1 = get_pdf(dist1, x)  # PDF for N(0, 1)
    pdf2 = get_pdf(dist2, x) 
    # Plot the PDFs
    plt.plot(x.numpy(), pdf1.numpy(), label=f'N({mean}, {std1 ** 2})')
    plt.plot(x.numpy(), pdf2.numpy(), label=f'N({mean}, {std2 ** 2})')
    plt.title('Normal Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Sample from the distribution and plot the histogram of the samples
    samples = dist1.sample((1000,))  # Draw 1000 samples
    get_distribution_viz(dist1, x)

    # %%
    # question 4c)
    # Create Y = 2X + 4, Plot the PDF of Y
    mean = 2
    std = 5
    dist3 = get_distribution(mean, std)
    samples = dist3.sample((1000,))  # Draw 1000 samples
    y = 2 * samples + 4  # Transform the samples

    # Compute the PDF of Y
    mean_y = 2 * mean + 4
    std_y = 2 * std
    dist_y = get_distribution(mean_y, std_y)
    x = torch.linspace(-42, 58, 1000)  # from -100 to 100 with 1000 points
    pdf_y = get_pdf(dist_y, x)

    get_distribution_viz(dist_y, x)


    # # visualize the y
    # plt.scatter(samples.numpy(), y.numpy(), alpha=0.5)
    # plt.title('Transformed Samples')
    # plt.xlabel('X samples')
    # plt.ylabel('Y samples')
    # plt.grid(True)
    # plt.show()
    



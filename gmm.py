from scipy.stats import norm
import numpy as np
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

"""
A Gaussian Mixture Model assumes that a datapoint stems from a combination
of individual gaussian distributions which are referred to as components.
Each component is assigned a weight that dictates how large the impact of the 
distribution to the overall distribution is. 
This class (the fit function to be precise) takes in a one dimensional input vector
which corresponds to x-values on a straight line and a value for K which is the number of components
it is supposed to learn. It then computes K normal distributions and their weights which best describe
the distribution of the data points on the x-axis. 

In theory you can extend this concept to multiple dimensions, however we do not.
"""
class GMM:
    def pdf(self, x, wgt, mu, sigma):
        """
        Computes the PDF of the Gaussian Mixture Model for one point or a vector of points.
        Parameters:
        x (float or np.ndarray): Point(s) where the PDF is evaluated.
        wgt (np.ndarray): Weights of the mixture components.
        mu (np.ndarray): Means of the components.
        sigma (np.ndarray): Standard deviations of the components.
        
        Returns:
        np.ndarray: The PDF values at x.
        """
        pdf_values = np.zeros_like(x, dtype=float) # initialize vector of 0s with length of x
        for k in range(len(wgt)): # iterate over components
            pdf_values += wgt[k] * norm.pdf(x, loc=mu[k], scale=sigma[k]) # compute weighted probability of each component
        return pdf_values
    
    def cdf(self, x, wgt, mu, sigma):
        """
        Computes the CDF of the Gaussian Mixture Model for one point or a vector of points.
        
        Parameters:
        x (float or np.ndarray): Points where the CDF is evaluated.
        wgt (np.ndarray): Weights of the mixture components.
        mu (np.ndarray): Means of the components.
        sigma (np.ndarray): Standard deviations of the components.
        
        Returns:
        np.ndarray: The CDF values at x.
        """
        cdf_values = np.zeros_like(x, dtype=float)
        for k in range(len(wgt)):
            cdf_values += wgt[k] * norm.cdf(x, loc=mu[k], scale=sigma[k])
        return cdf_values
    
    def rvs(self, wgt, mu, sigma, size=None, random_state=None):
        """
        Generate random samples from the Gaussian Mixture Model.
        
        Parameters:
        wgt (np.ndarray): Weights of the mixture components.
        mu (np.ndarray): Means of the components.
        sigma (np.ndarray): Standard deviations of the components.
        size (int): Number of samples to generate.
        random_state (int or np.random.Generator): Seed or random generator.
        
        Returns:
        np.ndarray: Generated random samples.
        """
        if random_state is None:
            random_state = np.random.default_rng()
        
        if size is None:
            size = 1
        
        components = random_state.choice(len(wgt), size=size, p=wgt)
        # Generate samples from the selected components
        samples = np.array([
            random_state.normal(loc=mu[k], scale=sigma[k]) for k in components
        ])
        
        return samples

    def fit(self, data, K):
        gmm = GaussianMixture(n_components=K, random_state=42)
        gmm.fit(data.reshape(-1, 1))  # Reshape to ensure compatibility
        
        wgt = gmm.weights_
        mu = gmm.means_.flatten()
        sigma = np.sqrt(gmm.covariances_.flatten())
        
        return {"weights": wgt, "means": mu, "std_devs": sigma}

if __name__ == '__main__':
    # Example data
    data = np.random.randn(1000)
    
    # Fit gmm model
    gmm = GMM()
    params = gmm.fit(data, K=5)
    print("Fitted parameters:", params)
    
    # Evaluate PDF and CDF
    x = np.linspace(-3, 3, 100)
    pdf_values = gmm.pdf(x, params['weights'], params['means'], params['std_devs'])
    cdf_values = gmm.cdf(x, params['weights'], params['means'], params['std_devs'])
    
    print("PDF values:", pdf_values[:5])
    print("CDF values:", cdf_values[:5])
    
    # Generate random samples
    samples = gmm.rvs(params['weights'], params['means'], params['std_devs'], size=10)
    print("Random samples:", samples)

    # Plot the GMM PDF
    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf_values, label="GMM PDF", color="blue", linewidth=2)

    # Overlay individual component PDFs
    for w, m, s in zip(params["weights"], params["means"], params["std_devs"]):
        plt.plot(x, w * norm.pdf(x, loc=m, scale=s), linestyle="--", label=f"Component μ={m}, σ={s}")

    # Add labels and legend
    plt.title("Gaussian Mixture Model PDF", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()


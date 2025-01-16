from gmm import GMM
import sys
import numpy as np

file_name = sys.argv[1]
#file_name = "data_file1.csv"

def string_to_number(string):
    """
    Splits the input string on ',', and converts everything to floats

    Parameters:
    string (str): comma separated float values

    Returns:
    np.ndarray: one or multidimensional array containing the processed data 

    """
    data = np.array([float(i) for i in line.split(",")])
    return data



def calc_llh(data, wgt, mu, sigma):
    """
    Compute the log-likelihood of the data given the Gaussian Mixture Model parameters.
    
    Parameters:
    data (np.ndarray): Data points to evaluate the likelihood for.
    wgt (np.ndarray): Weights of the mixture components.
    mu (np.ndarray): Means of the components.
    sigma (np.ndarray): Standard deviations of the components.
    
    Returns:
    float: Log-likelihood of the data.
    """
    gmm = GMM()
    pdf_values = gmm.pdf(data, wgt, mu, sigma)

    # Avoid log(0) by adding a small epsilon to the PDF values
    epsilon = 1e-15
    pdf_values = np.clip(pdf_values, epsilon, None)

    log_likelihood = np.sum(np.log(pdf_values))
    return log_likelihood

def calc_AIC(data, wgt, mu, sigma):
    """
    Compute the Akaike Information Criterion (AIC) for the Gaussian Mixture Model.
    It penelizes model compelxity by adding a term for the number of parameters.
    
    Parameters:
    data (np.ndarray): Data points used to fit the model.
    wgt (np.ndarray): Weights of the mixture components.
    mu (np.ndarray): Means of the components.
    sigma (np.ndarray): Standard deviations of the components.
    
    Returns:
    float: AIC value.
    """

    # Number of parameters in the model (k in the formula on Wikipedia)
    # For each component: weight, mean and variance  => 3 parameters per component
    num_params = 3 * len(wgt)
    
    aic = 2 * num_params - 2 * calc_llh(data, wgt, mu, sigma)
    return aic





if __name__ == '__main__':

    data = np.ndarray([])

    with open(file_name, "r") as f:
        lines=f.readlines()
        for line in lines:
            data = np.vstack(string_to_number(line))
    
    llhs = []
    AICs = []
    parameters = []

    for K in range(2,11):
        gmm = GMM()
        params = gmm.fit(data, K=K)
        weights = params['weights']
        means = params['means']
        sigma = params['std_devs']

        parameters.append(params)

        llh = calc_llh(data, weights, means, sigma)
        print(f"Log-likelihood for K={K}: {llh}")
        llhs.append(llh)

        AICs.append(calc_AIC(data, weights, means, sigma))

    weight=parameters[np.argmin(AICs)]['weights']
    mean=parameters[np.argmin(AICs)]['means']
    sigma=parameters[np.argmin(AICs)]['std_devs']
    print(f"The model with {int(np.argmin(AICs) + 2)} components has the best AIC value ({min(AICs)}). \n The models parameters are: weights={weight}, means={mean} and sigma={sigma}")

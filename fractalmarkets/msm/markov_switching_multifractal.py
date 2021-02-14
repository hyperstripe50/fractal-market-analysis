import numpy as np


class MarkovSwitchingMultifractal(object):

    def __init__(self, m_0, mu, sigma_bar, b, gamma_1, k_bar):
        assert 0. < m_0 <= 2.0, "m_0 must be within [0,2]"
        self.m_0 = m_0
        self.mu = mu
        self.standard_deviation_bar = sigma_bar
        assert b >= 1., "b must be greater than 1"
        self.b = b
        assert 0. < gamma_1 <= 1., "gamma_1 must be a probability"
        self.gamma_1 = gamma_1
        assert type(k_bar) == int and k_bar > 0, "k_bar must be an integer greater than 0"
        self.k_bar = k_bar

        # compute transition probabilities for each k:
        self.transition_probabilities = np.zeros(self.k_bar)
        self.transition_probabilities[0] = self.gamma_1
        for k in range(1, self.k_bar):
            gamma_k = self._get_transition_probability(k)
            self.transition_probabilities[k] = gamma_k
        # print('Transition probabilities = {}'.format(self.transition_probabilities))

        # initialise the state vector
        self.M = np.zeros(self.k_bar)
        for k in range(self.k_bar):
            self.M[k] = self._binomial_M()
        # print('Initial state vector = {}'.format(self.M))

    def _get_transition_probability(self, k):
        # gamma_k = 1 - (1 - self.gamma_1)**(self.b**(k-self.k_bar)) # this?
        gamma_k = 1 - (1 - self.gamma_1) ** (self.b ** (k - 1)) # or this?
        return gamma_k

    def _binomial_M(self):
        #TODO this could be a multinomial distribution instead
        if np.random.random() < 0.5:
            return self.m_0
        else:
            return 2. - self.m_0

    def _update_state_vector(self):
        for k in range(self.k_bar):
            if np.random.random() < self.transition_probabilities[k]:
                # draw M_k_t from the distribution for M
                self.M[k] = self._binomial_M()

    def timestep(self):
        # update state vector according to transition probabilities
        self._update_state_vector()
        # calculate this timesep's sigma value
        sigma = self.standard_deviation_bar * np.prod(self.M)
        r_t = sigma * np.random.normal(loc=0, scale=1)
        return r_t

    def simulate(self, timesteps=1000):
        # generate list of returns
        returns = np.zeros(timesteps)
        for t in range(timesteps):
            returns[t] = self.timestep()
        return returns

    def returns_to_prices(self, returns, same_size=True):
        # convert to raw prices
        prices = [self.mu] # is this a good choice of initialisation?
        for i, ret in enumerate(returns):
            p_t = prices[i] * np.exp(ret)
            prices.append(p_t)
        prices = np.array(prices)
        if same_size:
            prices = prices[1:]
        return prices


# demo
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    m_0 = 1.4
    mu = 0.1
    sigma_bar = 0.05
    b = 3.0
    gamma_1 = 0.3
    k_bar = 5

    # values from https://doc.sagemath.org/html/en/reference/finance/sage/finance/markov_multifractal.html
    # m_0 = 1.4
    # # sigma_bar = 0.5
    # b = 3.0
    # gamma_1 = 0.95
    # k_bar=3

    msm = MarkovSwitchingMultifractal(m_0=m_0, mu=mu, sigma_bar=sigma_bar, b=b, gamma_1=gamma_1, k_bar=k_bar)

    # run 10 simulations
    for _ in range(10):
        X = msm.simulate(timesteps=1000)
        P = msm.returns_to_prices(X)
        t = np.arange(0, len(X))
        plt.plot(t, X)
        plt.suptitle('MSM log returns')
        plt.show()
        plt.plot(t, P)
        plt.suptitle('MSM prices')
        plt.show()

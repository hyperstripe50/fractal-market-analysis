import numpy as np


class MarkovSwitchingMultifractal(object):

    def __init__(self, m_0=1.4, mu=0.1, sigma_bar=0.05, b=3.0, gamma_1=0.3, k_bar=5, timesteps=1000):
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
        assert type(timesteps) == int and timesteps > 1, "timesteps must be an integer greater than 1"
        self.timesteps = timesteps


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
        '''
        Compute the probabilities of transition between different states for each volatility frequency k
        :param k: the kth index
        :return: the probability of state transition at the kth level
        '''
        # I've seen some literature express this equation as:
        # gamma_k = 1 - (1 - self.gamma_1)**(self.b**(k - self.k_bar))

        # But Calvet & Fisher's original paper Regime-Switching and the Estimation of Multifractal Processes
        # expresses the following (Equation 2.2)
        gamma_k = 1 - (1 - self.gamma_1) ** (self.b ** (k - 1))
        return gamma_k

    def _binomial_M(self):
        '''
        The distribution from which values of M are drawn. Currently binomial but this could
        be extended to multinomial or lognormal.
        :return: A sample from the M distribution
        '''
        if np.random.random() < 0.5:
            return self.m_0
        else:
            return 2. - self.m_0

    def _update_state_vector(self):
        '''
        Using the given probabilities of state transition, update the state
        '''
        for k in range(self.k_bar):
            if np.random.random() < self.transition_probabilities[k]:
                # draw M_k_t from the distribution for M
                self.M[k] = self._binomial_M()

    def timestep(self):
        '''
        Run an update to the state vector and generate a new value of returns for the current timestep
        :return: r_t: float - the value of returns for timestep t
        '''
        # update state vector according to transition probabilities
        self._update_state_vector()
        # calculate this timesep's sigma value
        sigma = self.standard_deviation_bar * np.prod(self.M)
        r_t = sigma * np.random.normal(loc=0, scale=1)
        return r_t

    def simulate(self):
        returns = np.array([self.timestep() for _ in range(self.timesteps)])
        # The original paper models returns, so convert these to raw price values
        prices = self._returns_to_prices(returns)
        # join with an array of time values
        times = np.float32(np.arange(0, len(prices)))
        times /= len(times)
        return np.column_stack((times, prices))

    def _returns_to_prices(self, returns, same_size=True, unit_scale=True):
        '''
        Calvet & Fisher's paper models returns rather than raw prices, so convert the time series
        of returns to a time series of prices.
        :param returns: array of returns at each timestep
        :param same_size: Assuming a starting price of self.mu, the resulting price array would have one more element
        than the input returns array. Setting same_size=True will clip the first value of prices to make it equal length
        to the input array
        :return: corresponding array of prices at each timestep
        '''
        # convert to raw prices
        prices = [1.0] # is this a good choice of initialisation?
        for i, ret in enumerate(returns):
            p_t = prices[i] * np.exp(ret)
            prices.append(p_t)
        prices = np.array(prices)
        prices -= 1.0 # recentre starting price at 0 (because it needed a non-zero initialization)
        if same_size:
            prices = prices[1:]

        # rescale prices to unit interval?
        if unit_scale:
            min = np.min(prices)
            max = np.max(prices)
            range = max-min
            prices /= range

        return prices

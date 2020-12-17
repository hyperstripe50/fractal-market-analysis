import numpy as np

class MutiplicativeCascade:
    def __init__(self, k_max, M, randomize=False):
        self.k_max = k_max
        self.M = M
        self.randomize = randomize
        self.data = []
    
    def cascade(self):
        y = self._cascade_recursively(1, 1, 1, self.k_max, self.M, self.randomize)
        x = np.linspace(0, 1, num=len(y), endpoint=False)

        y = np.insert(y, 0, 0)
        x = np.append(x, 1)

        self.data = np.stack([x, y], axis=1)

    def _cascade_recursively(self, x, y, k, k_max, M, randomize=False):
        """
        :param x: width of current cell
        :param y: height of current cell
        :param k: current branch of the recursion tree
        :param k_max: max depth of the recursion tree
        :param M: array m0, m1, ..., mb where sum(M) = 1
        :param randomize: whether or not to shuffle M before assigning mass to child cells. See page 13 of "A Multifractal Model of Asset Returns" 1997
        :return: [y, ...] corresponding to multiplicative cascade y coordinates
        """
        a = x * y
        x_next = x / len(M)

        M_shuffle = np.copy(M)
        if randomize:
            np.random.shuffle(M_shuffle)
        else:
            M_shuffle = M_shuffle
            
        y_i = np.array([])
        if (k == k_max):
            for m in M_shuffle:
                y_i = np.append(y_i, (m * a) / x_next)

            return y_i

        for m in M_shuffle:
            y_i = np.append(y_i, self._cascade_recursively(x_next, (m * a) / x_next, k + 1, k_max, M, randomize))
        
        return y_i
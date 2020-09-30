import numpy as np

class BrownianMotion:

    def __init__(self, k_max, x, y, randomize=False):
        """
        y^(1/H) = x
        :param k_max: max depth of the recursion tree
        :param x: x coord of point P in generator. Brownian motion x=4/9 if y=2/3
        :param y: y coord of point P in generator. Brownian motion x=4/9 if y=2/3
        :param randomize: randomize symmetric generator segments
        """
        self.k_max = k_max
        self.x = x
        self.y = y
        self.randomize = randomize

    def simulate(self):
        """
        :return: x, y of fbm timeseries
        """

        fbm = np.array(self._simulate_bm_recursively(0, 0, 1, 1, 1, self.k_max, self.x, self.y, randomize=self.randomize))
        x = fbm[:,0]
        y = fbm[:,1]
        x = np.delete(x, np.arange(0, x.size, 4))
        y = np.delete(y, np.arange(0, y.size, 4))

        x = np.insert(x, 0, 0)
        y = np.insert(y, 0, 0)

        return x, y

    def _simulate_bm_recursively(self, x1, y1, x2, y2, k, k_max, x, y, cdf=None, randomize=True):
        """
        H = 1/2
             ________
        2/3 |__     /|
            |  /\  / | 1
            | /| \/__| 1/3
            |/_|__|__|
            4/9  5/9
                1

        y^(1/H) + (2y - 1)^(1/H) + y^(1/H) = 1
        y = x^H

        :param x1: left x coord of initiator
        :param y1: left y coord of initiator
        :param x2: right x coord of initiator
        :param y2: right y coord of initiator
        :param k: current branch of the recursion tree
        :param k_max: max depth of the recursion tree
        :param x: x coord of point P in generator. Brownian motion x=4/9 if y=2/3
        :param y: y coord of point P in generator. Brownian motion x=4/9 if y=2/3
        :param cdf: cdf of trading time
        :return: simulated time series for fractional brownian motion
        """
        p0, p1, p2, p3 = self._construct_generator_from_initiator(x1, y1, x2, y2, x, y)

        if cdf != None:
            p0, p1, p2, p3 = self._deform_clock_time(p0, p1, p2, p3, cdf)

        if randomize:
            p0, p1, p2, p3 = self._randomize_generator_segments(p0, p1, p2, p3)

        if (k == k_max):
            return [p0, p1, p2, p3]

        fbm = self._simulate_bm_recursively(p0[0], p0[1], p1[0], p1[1], k+1, k_max, x, y, cdf, randomize)
        fbm = np.append(fbm, self._simulate_bm_recursively(p1[0], p1[1], p2[0], p2[1], k+1, k_max, x, y, cdf, randomize), axis=0)
        fbm = np.append(fbm, self._simulate_bm_recursively(p2[0], p2[1], p3[0], p3[1], k+1, k_max, x, y, cdf, randomize), axis=0)

        return fbm

    def _construct_generator_from_initiator(self, x1, y1, x2, y2, x, y):
        """
              _______ (x2, y2)
             |      /|
             |    /  |
             |  /    |
             |/_ __ _|
        (x1, y1)

             Initiator
                |
                |
             Generator
             ________
        2/3 |__     /|
            |  /\  / | 1
            | /| \/__| 1/3
            |/_|__|__|
            4/9  5/9
                1

        y^(1/H) + (2y - 1)^(1/H) + y^(1/H) = 1
        y = x^H

        :param x1: left x coord of initiator
        :param y1: left y coord of initiator
        :param x2: right x coord of initiator
        :param y2: right y coord of initiator
        :param x: x coord of point P in generator. Brownian motion x=4/9 if y=2/3
        :param y: y coord of point P in generator. Brownian motion x=4/9 if y=2/3
        :return: three segment symmetric generator
        """
        delta_x = x2 - x1
        delta_y = y2 - y1

        p0 = [x1, y1]
        p1 = [x1 + delta_x * x, y1 + delta_y * y]
        p2 = [p1[0] + delta_x * (1 - 2 * x), p1[1] - delta_y * ((2 * y) - 1)]
        p3 = [x2, y2]

        return p0, p1, p2, p3

    def _deform_clock_time(self, p0, p1, p2, p3, cdf):
        """
        Deforms clock time into trading time as defined by the CDF of a multiplicative cascade. See chapter 11 of Misbehavior of Markets.
        :param p0:  point 0 of generator
        :param p1:  point 1 of generator
        :param p2:  point 2 of generator
        :param p3:  point 3 of generator
        :param cdf: Trading Time CDF interp1d function
        :return: new x,y coordinates of all points after deforming time
        """
        x1 = p0[0]
        x2 = p3[0]

        dT1 = cdf(p1[0]) - cdf(p0[0])
        dT2 = cdf(p2[0]) - cdf(p1[0])
        dT3 = cdf(p3[0]) - cdf(p2[0])
        p1[0] = p0[0] + (x2 - x1) * (dT1 / (dT1 + dT2 + dT3))
        p2[0] = p1[0] + (x2 - x1) * (dT2 / (dT1 + dT2 + dT3))
        p3[0] = p2[0] + (x2 - x1) * (dT3 / (dT1 + dT2 + dT3))

        return p0, p1, p2, p3

    def _randomize_generator_segments(self, p0, p1, p2, p3):
        """
            ________p3
            |  p1   /|
            |  /\  / | 1
            | /  \/  |
            |/_ _p2__|
           p0    1

          Not Randomized
                |
                |
            Randomized

                  /\
             ____/__\
            |   /    |
            |  /     | 1
            | /      |
            |/_ __ __|
                1

        Randomizes the segments of a generator without rotation.
        :param p0: left-most coordinates of the generator
        :param p1: coordinates of the first break of the generator
        :param p2: coordinates of the second break of the generator
        :param p3: coordinates of the right-most point of the generator
        :return: reordered generator without rotation.
        """
        w0 = p1[0] - p0[0]
        h0 = p1[1] - p0[1]

        w1 = p2[0] - p1[0]
        h1 = p2[1] - p1[1]

        w2 = p3[0] - p2[0]
        h2 = p3[1] - p2[1]

        segments = [[w0, h0], [w1, h1], [w2, h2]]
        np.random.shuffle(segments)

        x_0 = p0[0]
        y_0 = p0[1]

        x_1 = x_0 + segments[0][0]
        y_1 = y_0 + segments[0][1]

        x_2 = x_1 + segments[1][0]
        y_2 = y_1 + segments[1][1]

        x_3 = x_2 + segments[2][0]
        y_3 = y_2 + segments[2][1]

        p0 = [x_0, y_0]
        p1 = [x_1, y_1]
        p2 = [x_2, y_2]
        p3 = [x_3, y_3]

        return p0, p1, p2, p3
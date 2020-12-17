from fractalmarkets.rs.rs import RS
from fractalmarkets.mmar.brownian_motion_multifractal_time import BrownianMotionMultifractalTime

bmmt = BrownianMotionMultifractalTime(9, x=4/9, y=0.603, randomize_segments=False, randomize_time=False, M=[0.6, 0.4])
data = bmmt.simulate()

rs = RS(data[1:,1]) # timeseries starts at zero which must be omitted to avoid division error
(H, c) = rs.get_Hc()
print("Estimated H from RS Analysis: {}".format(H))

rs.plot_vstat() # plot vstat and RS
def perfect_reward(number_outage):
    if number_outage == 0:
        return 1
    else:
        return 0

def initial_reward(meanSINR, nOut, nUE):
    reward = (meanSINR / 20) + (-1.0 * nOut /nUE)
    return max(reward, -1)

def outage_reward(nOut, nUE):
    return (nUE - nOut) / nUE

def sinr_capped(mean_capped_sinr, outage_threshold):
    return mean_capped_sinr / outage_threshold

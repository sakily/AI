import numpy as np
import pandas as pd


def calc_slope(data: pd.DataFrame):
    '''order slopr factor'''
    return ((np.log(data['a1']) - np.log(data['b1'])) / (np.log(data['a1_v']) - np.log(data['b1_v']))).replace([-np.inf, np.inf], np.nan)

def calc_soir(data: pd.DataFrame):
    '''step order imbalance ratio'''
    numerator_ = 0
    w = []
    for i in range(1,6):
        numerator = data['b'+str(i)+'_v']-data['a'+str(i)+'_v']
        denominator = data['b'+str(i)+'_v']+data['a'+str(i)+'_v']
        # data['SOIR_'+str(i)] = (data['b'+str(i)+'_v']-data['a'+str(i)+'_v']) / (data['b'+str(i)+'_v']+data['a'+str(i)+'_v'])
        data['SOIR_' + str(i)] = numerator/denominator
        w.append(1-(i-1)/5)
        numerator_ += data['SOIR_'+str(i)] * w[i-1]

    weight_sum = np.sum(w)
    data['SOIR'] = numerator_ /weight_sum

    return data

def calc_pres(data: pd.DataFrame):
    """calculate buy&sell imbalance factor"""

    bench_price = data['last'].iloc[-1]
    _ = np.arange(1, 6)
    bid_d = [bench_price / (bench_price - data["b%s" % s]) for s in _]
    # bid_d = [_.replace(np.inf,0) for _ in bid_d]
    bid_denominator = sum(bid_d)

    bid_weights = [(d / bid_denominator).replace(np.nan,1) for d in bid_d]

    press_buy = sum([data["b%s_v" % (i + 1)] * w for i, w in enumerate(bid_weights)])

    ask_d = [bench_price / (data['a%s' % s] - bench_price) for s in _]
    # ask_d = [_.replace(np.inf,0) for _ in ask_d]
    ask_denominator = sum(ask_d)

    ask_weights = [d / ask_denominator for d in ask_d]

    press_sell = sum([data['a%s_v' % (i + 1)] * w for i, w in enumerate(ask_weights)])

    return (np.log(press_buy) - np.log(press_sell)).replace([-np.inf, np.inf], np.nan)
    # return (np.log(press_buy) - np.log(press_sell))

def utils(data: pd.DataFrame):
    '''calculate the factors'''
    data['slope'] = calc_slope(data)
    data['pressure'] = calc_pres(data)
    data = calc_soir(data)
    return data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import savetxt
from scipy.interpolate import UnivariateSpline


epsilons_ratio = [0.01, 0.01, 0.01, 0.0105, 0.011, 0.0115, 0.012, 0.0125, 0.013, 0.0135, 0.014, 0.0145, 0.015, 0.0155,
                  0.016, 0.0165, 0.017, 0.0175, 0.018, 0.0185, 0.019, 0.0195, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025]

alpha = ['1_12', '1_11', '1_1', '105_1', '11_1', '115_1', '12_1', '125_1', '13_1', '135_1', '14_1', '145_1', '15_1',
         '155_1', '16_1', '165_1', '17_1', '175_1', '18_1', '185_1', '19_1', '195_1', '2_1', '21_1', '22_1', '23_1',
         '24_1', '25_1']

column_names = ["alpha", "epsilon", "accuracy", "index"]

result = pd.DataFrame(columns = column_names)
#13 21
for i in range(len(alpha)):
    if i == 23 or i == 24 or i == 25 or i == 26 or i == 27:
        continue
    df = pd.read_csv('wha_'+str(alpha[i])+'_equal_opportunity.csv')
    df_acc = pd.read_csv('wha_'+str(alpha[i])+'_accuracy_in_all_iterations.csv')

    if i == 16:
        df = pd.read_csv('wha_' + str(alpha[i]) + '_6_equal_opportunity.csv')
        df_acc = pd.read_csv('wha_' + str(alpha[i]) + '_6_accuracy_in_all_iterations.csv')

    if i == 17:
        df = pd.read_csv('wha_' + str(alpha[i]) + '_10_equal_opportunity.csv')
        df_acc = pd.read_csv('wha_' + str(alpha[i]) + '_10_accuracy_in_all_iterations.csv')

    df = df.drop(columns='Unnamed: 0')
    df_acc = df_acc.drop(columns='Unnamed: 0')
    datas = np.array(df)
    datas_acc = np.array(df_acc)
    df = []
    for each in datas:
        df.append(each[0])
    df_acc = []
    for each in datas_acc:
        df_acc.append(each[0])
    eps = epsilons_ratio[i]
    if i in [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
        epsilons1 = [eps * x for x in range(3000)]
    else:
        epsilons1 = [eps * x for x in range(2000)]
    epsilons1[0] = 0.0001
    last_eps_value = epsilons1[-1]
    if i in [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
        xp1 = np.linspace(0, last_eps_value, 3000)
    else:
        xp1 = np.linspace(0, last_eps_value, 2000)
    #xp1 = np.linspace(0, last_eps_value, 3000)
    z1 = np.polyfit(epsilons1, df, 5)
    p1 = np.poly1d(z1)
    view = p1(xp1)
    #spl = UnivariateSpline(epsilons1, df, w=None, bbox=[None, None], k=5, s=None, ext=0, check_finite=False)

    #plt.plot(xp1, df)
    #plt.plot(xp1, spl(xp1), 'g', lw=5)
    #plt.plot(xp1, p1(xp1))
    #plt.show()
    #asign = np.sign(view)
    #index_changed_1 = np.where(asign == 1)
    savetxt('data.csv', datas, delimiter=',')
    savetxt('intrpolate.csv', view, delimiter=',')
    zero_crossings = np.where(np.diff(np.sign(view)))[0]
    if i == 2:
        try:
            index_changed = zero_crossings[1]
        except:
            index_changed = 0
    else:
        try:
            index_changed = zero_crossings[0]
        except:
            index_changed = 0
    #if i == 2:
        #index_changed = index_changed_1[0][1]
    #else:
        #index_changed = index_changed_1[0][0]

    epsilon_value = epsilons1[index_changed]
    acc = df_acc[index_changed]
    #try:

    #except:
    #    index_changed = 2000
    #    epsilon_value = 0
    #    acc = 0
    #if index_changed != 0:

    #if index_changed != 0 and index_changed!=11:
    result.loc[len(result.index)]= [float(eps)/float(0.01), epsilon_value, acc, int(index_changed)]
    print ('hi')
result.to_csv('wha_3D_again_3.csv')


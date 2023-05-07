import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

#epsilons_ratio = [0.01, 0.0102, 0.0105, 0.0107, 0.0108, 0.011, 0.015, 0.02, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08]

#alpha = ["same", "102_1", "105_1", "107_1", "108_1", "11_1", "15_1", "2_1", "3_1", "35_1", "4_1", "5_1", "6_1", "8_1"]

epsilons_ratio = [0.01, 0.0105, 0.011, 0.0115, 0.012, 0.0125, 0.013, 0.0135, 0.014, 0.0145, 0.015, 0.0155, 0.016, 0.0165,
                  0.017, 0.0175, 0.018, 0.0185, 0.019, 0.0195, 0.02, 0.0205, 0.021, 0.0215, 0.022, 0.0225, 0.023, 0.0235,
                  0.024, 0.0245,  0.025,  0.0255,  0.026, 0.0265, 0.027, 0.0275, 0.028, 0.0285, 0.029, 0.0295, 0.03]

alpha = ["same", "105_1", "11_1", "115_1", "12_1", "125_1", "13_1", "135_1", "14_1", "145_1", "15_1", "155_1", "16_1",
          "165_1", "17_1", "175_1", "18_1", "185_1", "19_1", "195_1", "2_1", "205_1", "21_1", "215_1", "22_1", "225_1",
          "23_1", "235_1", "24_1", "245_1", "25_1", "255_1", "26_1", "265_1", "27_1", "275_1", "28_1", "285_1", "29_1",
          "295_1", "3_1"]

column_names = ["alpha", "epsilon", "accuracy", "index"]

result = pd.DataFrame(columns = column_names)

for i in range(len(alpha)):
    df = pd.read_csv('lsat_wb_'+str(alpha[i])+'_equal_opportunity_1.csv')
    df_acc = pd.read_csv('lsat_wb_'+str(alpha[i])+'_accuracy_in_all_iterations_1.csv')

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
    epsilons1 = [eps * x for x in range(2000)]
    epsilons1[0] = 0.0001
    last_eps_value = epsilons1[-1]
    xp1 = np.linspace(0, last_eps_value, 2000)
    z1 = np.polyfit(epsilons1, df, 3)
    p1 = np.poly1d(z1)
    view = p1(xp1)
    spl = UnivariateSpline(epsilons1, df, w=None, bbox=[None, None], k=5, s=None, ext=0, check_finite=False)

    #plt.plot(xp1, df)
    #plt.plot(xp1, spl(xp1), 'g', lw=5)
    #plt.plot(xp1, p1(xp1))
    #plt.show()
    asign = np.sign(view)
    try:
        index_changed = np.where(asign == 1)[0][0]
        epsilon_value = epsilons1[index_changed]
        acc = df_acc[index_changed]
    except:
        index_changed = 0
        epsilon_value = 0
        acc = 0

    if index_changed != 0:
        result.loc[len(result.index)] = [float(eps)/float(0.01), epsilon_value, acc, int(index_changed)]
    print ('hi')
result.to_csv('lsat_3D_2.csv')


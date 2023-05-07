import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

epsilons_ratio = [0.01, 0.01, 0.01, 0.0105, 0.011, 0.0115, 0.012, 0.0125, 0.013, 0.0135, 0.014, 0.0145, 0.015, 0.0155, 0.016,
                  0.0165, 0.017, 0.0175, 0.018, 0.0185, 0.019, 0.0195, 0.02, 0.0205, 0.021, 0.0215, 0.022, 0.0225,
                  0.023, 0.0235, 0.024, 0.0245, 0.025, 0.0255, 0.026, 0.0265, 0.027, 0.0275, 0.028, 0.0285, 0.029,
                  0.0295, 0.03, 0.0305, 0.031, 0.0315, 0.032, 0.0325, 0.033, 0.0335, 0.034, 0.0345, 0.035, 0.0355,
                  0.036, 0.0365, 0.037, 0.0375, 0.038, 0.0385, 0.039, 0.0395, 0.04, 0.0405, 0.041, 0.0415, 0.042,
                  0.0425, 0.043, 0.0435, 0.044, 0.0445, 0.045, 0.0455, 0.046, 0.0465, 0.047, 0.0475, 0.048, 0.0485,
                  0.049, 0.0495, 0.05, 0.0505, 0.051, 0.0515, 0.052, 0.0525, 0.053, 0.0535, 0.054, 0.0545, 0.055,
                  0.0555, 0.056, 0.0565, 0.057, 0.0575, 0.058, 0.0585, 0.059, 0.0595,]

alpha = ['1_11', '1_105', '1_1', '105_1', '11_1', '115_1', '12_1', '125_1', '13_1', '135_1',
         '14_1', '145_1', '15_1', '155_1', '16_1', '165_1', '17_1', '175_1', '18_1', '185_1',
         '19_1', '195_1', '2_1', '205_1', '21_1', '215_1', '22_1', '225_1', '23_1', '235_1',
         '24_1', '245_1', '25_1', '255_1', '26_1', '265_1', '27_1', '275_1', '28_1', '285_1',
         '29_1', '295_1', '3_1', '305_1', '31_1', '315_1', '32_1', '325_1', '33_1', '335_1',
         '34_1', '345_1', '35_1', '355_1', '36_1', '365_1', '37_1', '375_1', '38_1', '385_1',
         '39_1', '395_1', '4_1', '405_1', '41_1', '415_1', '42_1', '425_1', '43_1', '435_1',
         '44_1', '445_1', '45_1', '455_1', '46_1', '465_1', '47_1', '475_1', '48_1', '485_1',
         '49_1', '495_1', '5_1', '505_1', '51_1', '515_1', '52_1', '525_1', '53_1', '535_1',
         '54_1', '545_1', '55_1', '555_1', '56_1', '565_1', '57_1', '575_1', '58_1', '585_1',
         '59_1', '595_1']

column_names = ["alpha", "epsilon", "accuracy", "index"]

result = pd.DataFrame(columns = column_names)

#alpha = ['10000_1']
#epsilons_ratio = [1]
for i in range(len(alpha)):
    df = pd.read_csv('wb_'+str(alpha[i])+'_equal_opportunity.csv')
    df_acc = pd.read_csv('wb_'+str(alpha[i])+'_accuracy_in_all_iterations.csv')

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
    asign = np.sign(view)
    index_changed = np.where(asign == 1)[0][0]
    #if index_changed != 0:
    epsilon_value = epsilons1[index_changed]
    acc = df_acc[index_changed]
    #if index_changed != 0 and index_changed!=11:
    result.loc[len(result.index)]= [float(eps)/float(0.01), epsilon_value, acc, int(index_changed)]
    print ('hi')
result.to_csv('wb_3D.csv')


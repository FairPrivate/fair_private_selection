import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

epsilons1 = [0.01*x for x in range(2000)]
epsilons1[0] = 0.0001

epsilons2 = [0.0102*x for x in range(2000)]
epsilons2[0] = 0.0001

epsilons3 = [0.0105*x for x in range(2000)]
epsilons3[0] = 0.0001

epsilons4 = [0.0107*x for x in range(2000)]
epsilons4[0] = 0.0001

epsilons5 = [0.0108*x for x in range(2000)]
epsilons5[0] = 0.0001

epsilons6 = [0.011*x for x in range(2000)]
epsilons6[0] = 0.0001

epsilons7 = [0.015*x for x in range(2000)]
epsilons7[0] = 0.0001

epsilons8 = [0.02*x for x in range(2000)]
epsilons8[0] = 0.0001

epsilons9 = [0.03*x for x in range(2000)]
epsilons9[0] = 0.0001

epsilons10 = [0.035*x for x in range(2000)]
epsilons10[0] = 0.0001

epsilons11 = [0.04*x for x in range(2000)]
epsilons11[0] = 0.0001

epsilons12 = [0.05*x for x in range(2000)]
epsilons12[0] = 0.0001

epsilons13 = [0.06*x for x in range(2000)]
epsilons13[0] = 0.0001

epsilons14 = [0.08*x for x in range(2000)]
epsilons14[0] = 0.0001

epsilons15 = [0.1*x for x in range(200)]
epsilons15[0] = 0.0001

names = ['lsat_wb_prob_a', 'lsat_wb_prob_b', 'lsat_wb_equal_opportunity', 'lsat_wb_accuracy_in_all_iterations', 'lsat_wb_accuracy_in_all',
         'lsat_wb_a_accuracy_in_iterations', 'lsat_wb_b_accuracy_in_iterations']


lsat_wb_file_name_102_1 = ['lsat_wb_102_1_prob_a', 'lsat_wb_102_1_prob_b', 'lsat_wb_102_1_equal_opportunity', 'lsat_wb_102_1_accuracy_in_all',
                    'lsat_wb_102_1_accuracy_in_all_iterations', 'lsat_wb_102_1_accuracy_in_a_iterations',
                    'lsat_wb_102_1_accuracy_in_b_iterations']

lsat_wb_file_name_105_1 = ['lsat_wb_105_1_prob_a', 'lsat_wb_105_1_prob_b', 'lsat_wb_105_1_equal_opportunity', 'lsat_wb_105_1_accuracy_in_all',
                      'lsat_wb_105_1_accuracy_in_all_iterations', 'lsat_wb_105_1_accuracy_in_a_iterations',
                      'lsat_wb_105_1_accuracy_in_b_iterations']

lsat_wb_file_name_107_1 = ['lsat_wb_107_1_prob_a', 'lsat_wb_107_1_prob_b', 'lsat_wb_107_1_equal_opportunity', 'lsat_wb_107_1_accuracy_in_all',
                      'lsat_wb_107_1_accuracy_in_all_iterations', 'lsat_wb_107_1_accuracy_in_a_iterations',
                      'lsat_wb_107_1_accuracy_in_b_iterations']

lsat_wb_file_name_1_1 = ['lsat_wb_same_prob_a', 'lsat_wb_same_prob_b', 'lsat_wb_same_equal_opportunity', 'lsat_wb_same_accuracy_in_all',
                    'lsat_wb_same_accuracy_in_all_iterations', 'lsat_wb_same_accuracy_in_a_iterations',
                    'lsat_wb_same_accuracy_in_b_iterations']

lsat_wb_file_name_108_1 = ['lsat_wb_108_1_prob_a', 'lsat_wb_108_1_prob_b', 'lsat_wb_108_1_equal_opportunity', 'lsat_wb_108_1_accuracy_in_all',
                      'lsat_wb_108_1_accuracy_in_all_iterations', 'lsat_wb_108_1_accuracy_in_a_iterations',
                      'lsat_wb_108_1_accuracy_in_b_iterations']

lsat_wb_file_name_11_1 = ['lsat_wb_11_1_prob_a', 'lsat_wb_11_1_prob_b', 'lsat_wb_11_1_equal_opportunity', 'lsat_wb_11_1_accuracy_in_all',
                      'lsat_wb_11_1_accuracy_in_all_iterations', 'lsat_wb_11_1_accuracy_in_a_iterations',
                      'lsat_wb_11_1_accuracy_in_b_iterations']

lsat_wb_file_name_15_1 = ['lsat_wb_15_1_prob_a', 'lsat_wb_15_1_prob_b', 'lsat_wb_15_1_equal_opportunity', 'lsat_wb_15_1_accuracy_in_all',
                      'lsat_wb_15_1_accuracy_in_all_iterations', 'lsat_wb_15_1_accuracy_in_a_iterations',
                      'lsat_wb_15_1_accuracy_in_b_iterations']

lsat_wb_file_name_2_1 = ['lsat_wb_2_1_prob_a', 'lsat_wb_2_1_prob_b', 'lsat_wb_2_1_equal_opportunity', 'lsat_wb_2_1_accuracy_in_all',
                      'lsat_wb_2_1_accuracy_in_all_iterations', 'lsat_wb_2_1_accuracy_in_a_iterations',
                      'lsat_wb_2_1_accuracy_in_b_iterations']

lsat_wb_file_name_3_1 = ['lsat_wb_3_1_prob_a', 'lsat_wb_3_1_prob_b', 'lsat_wb_3_1_equal_opportunity', 'lsat_wb_3_1_accuracy_in_all',
                     'lsat_wb_3_1_accuracy_in_all_iterations', 'lsat_wb_3_1_accuracy_in_a_iterations',
                     'lsat_wb_3_1_accuracy_in_b_iterations']


lsat_wb_file_name_35_1 = ['lsat_wb_35_1_prob_a', 'lsat_wb_35_1_prob_b', 'lsat_wb_35_1_equal_opportunity', 'lsat_wb_35_1_accuracy_in_all',
                     'lsat_wb_35_1_accuracy_in_all_iterations', 'lsat_wb_35_1_accuracy_in_a_iterations',
                     'lsat_wb_35_1_accuracy_in_b_iterations']

lsat_wb_file_name_4_1 = ['lsat_wb_4_1_prob_a', 'lsat_wb_4_1_prob_b', 'lsat_wb_4_1_equal_opportunity', 'lsat_wb_4_1_accuracy_in_all',
                    'lsat_wb_4_1_accuracy_in_all_iterations', 'lsat_wb_4_1_accuracy_in_a_iterations',
                    'lsat_wb_4_1_accuracy_in_b_iterations']

lsat_wb_file_name_5_1 = ['lsat_wb_5_1_prob_a', 'lsat_wb_5_1_prob_b', 'lsat_wb_5_1_equal_opportunity', 'lsat_wb_5_1_accuracy_in_all',
                    'lsat_wb_5_1_accuracy_in_all_iterations', 'lsat_wb_5_1_accuracy_in_a_iterations',
                    'lsat_wb_5_1_accuracy_in_b_iterations']

lsat_wb_file_name_6_1 = ['lsat_wb_6_1_prob_a', 'lsat_wb_6_1_prob_b', 'lsat_wb_6_1_equal_opportunity', 'lsat_wb_6_1_accuracy_in_all',
                    'lsat_wb_6_1_accuracy_in_all_iterations', 'lsat_wb_6_1_accuracy_in_a_iterations',
                    'lsat_wb_6_1_accuracy_in_b_iterations']

lsat_wb_file_name_8_1 = ['lsat_wb_8_1_prob_a', 'lsat_wb_8_1_prob_b', 'lsat_wb_8_1_equal_opportunity', 'lsat_wb_8_1_accuracy_in_all',
                    'lsat_wb_8_1_accuracy_in_all_iterations', 'lsat_wb_8_1_accuracy_in_a_iterations',
                    'lsat_wb_8_1_accuracy_in_b_iterations']

'''
debiased = ['lsat_wb_prob_a_debiased_multiple_selection', 'lsat_wb_prob_b_debiased_multiple_selection',
            'lsat_wb_equal_opportunity_debiased_multiple_selection', 'lsat_wb_accuracy_debiased_multiple_selection',
                    'lsat_wb_accuracy_in_all_iterations_debiased_multiple_selection',
            'lsat_wb_accuracy_in_a_iterations_debiased_multiple_selection',
                    'lsat_wb_accuracy_in_b_iterations_debiased_multiple_selection']

'''

debiased = ['lsat_wb_equal_opportunity_debiased_multiple_selection_1_debiased_eo_7_3_50_2', 'lsat_wb_equal_opportunity_debiased_multiple_selection_1_debiased_eo_7_3_50_2',
            'lsat_wb_equal_opportunity_debiased_multiple_selection_1_debiased_eo_7_3_50_2', 'lsat_wb_accuracy_in_all_iterations_debiased_multiple_selection_1_debiased_eo_7_3_50_2',
                    'lsat_wb_accuracy_in_all_iterations_debiased_multiple_selection_1_debiased_eo_7_3_50_2',
            'lsat_wb_accuracy_in_all_iterations_debiased_multiple_selection_1_debiased_eo_7_3_50_2',
                    'lsat_wb_accuracy_in_all_iterations_debiased_multiple_selection_1_debiased_eo_7_3_50_2']


'''
same_noise_lsat_wb_file_name = ['lsat_wb_same_prob_a', 'lsat_wb_same_prob_b', 'lsat_wb_same_equal_opportunity', 'lsat_wb_same_accuracy_in_all',
                           'lsat_wb_same_accuracy_in_all_iterations', 'lsat_wb_same_accuracy_in_a_iterations',
                           'lsat_wb_same_accuracy_in_b_iterations']
'''

expeiments = [ lsat_wb_file_name_1_1, lsat_wb_file_name_102_1, lsat_wb_file_name_105_1,
               lsat_wb_file_name_107_1, lsat_wb_file_name_108_1, lsat_wb_file_name_11_1,
              lsat_wb_file_name_15_1, lsat_wb_file_name_2_1, lsat_wb_file_name_3_1,
               lsat_wb_file_name_35_1, lsat_wb_file_name_4_1, lsat_wb_file_name_5_1,
               lsat_wb_file_name_6_1, lsat_wb_file_name_8_1, debiased]
proportions = [[1, 1], [1.02, 1], [1.05, 1], [1.07, 1], [1.08, 1], [1.1, 1], [1.5, 1]
               , [2, 1], [3, 1], [3.5, 1], [4, 1], [5, 1], [6, 1], [8, 1], [1, 1]]
for name_index in range(len(lsat_wb_file_name_1_1)):
    data1_1 = pd.read_csv(str(expeiments[0][name_index])+'_1.csv')

    data2_1 = pd.read_csv(str(expeiments[1][name_index])+'_1.csv')

    data3_1 = pd.read_csv(str(expeiments[2][name_index])+'_1.csv')

    data4_1 = pd.read_csv(str(expeiments[3][name_index])+'_1.csv')

    data5_1 = pd.read_csv(str(expeiments[4][name_index])+'_1.csv')

    data6_1 = pd.read_csv(str(expeiments[5][name_index])+'_1.csv')

    data7_1 = pd.read_csv(str(expeiments[6][name_index])+'_1.csv')

    data8_1 = pd.read_csv(str(expeiments[7][name_index])+'_1.csv')

    data9_1 = pd.read_csv(str(expeiments[8][name_index])+'_1.csv')

    data10_1 = pd.read_csv(str(expeiments[9][name_index]) + '_1.csv')

    data11_1 = pd.read_csv(str(expeiments[10][name_index]) + '_1.csv')

    data12_1 = pd.read_csv(str(expeiments[11][name_index]) + '_1.csv')

    data13_1 = pd.read_csv(str(expeiments[12][name_index]) + '_1.csv')

    data14_1 = pd.read_csv(str(expeiments[13][name_index]) + '_1.csv')

    data15_1 = pd.read_csv(str(expeiments[14][name_index]) + '.csv')

    data1_1 = data1_1.drop(columns='Unnamed: 0')
    data2_1 = data2_1.drop(columns='Unnamed: 0')
    data3_1 = data3_1.drop(columns='Unnamed: 0')
    data4_1 = data4_1.drop(columns='Unnamed: 0')
    data5_1 = data5_1.drop(columns='Unnamed: 0')
    data6_1 = data6_1.drop(columns='Unnamed: 0')
    data7_1 = data7_1.drop(columns='Unnamed: 0')
    data8_1 = data8_1.drop(columns='Unnamed: 0')
    data9_1 = data9_1.drop(columns='Unnamed: 0')
    data10_1 = data10_1.drop(columns='Unnamed: 0')
    data11_1 = data11_1.drop(columns='Unnamed: 0')
    data12_1 = data12_1.drop(columns='Unnamed: 0')
    data13_1 = data13_1.drop(columns='Unnamed: 0')
    data14_1 = data14_1.drop(columns='Unnamed: 0')
    data15_1 = data15_1.drop(columns='Unnamed: 0')


    datas = [data1_1, data2_1, data3_1, data4_1, data5_1, data6_1, data7_1, data8_1, data9_1, data10_1, data11_1,
             data12_1, data13_1, data14_1, data15_1]

    for data_index in range(len(datas)):
        data11 = np.array(datas[data_index])
        datas[data_index] = []
        for each in data11:
            datas[data_index].append(each[0])

    data1_1 = datas[0]
    data2_1 = datas[1]
    data3_1 = datas[2]
    data4_1 = datas[3]
    data5_1 = datas[4]
    data6_1 = datas[5]
    data7_1 = datas[6]
    data8_1 = datas[7]
    data9_1 = datas[8]
    data10_1 = datas[9]
    data11_1 = datas[10]
    data12_1 = datas[11]
    data13_1 = datas[12]
    data14_1 = datas[13]
    data15_1 = datas[14]

    data2_1 = data2_1[:1961]
    epsilons2 = epsilons2[:1961]

    data3_1 = data3_1[:1905]
    epsilons3 = epsilons3[:1905]

    data4_1 = data4_1[:1870]
    epsilons4 = epsilons4[:1870]

    data5_1 = data5_1[:1852]
    epsilons5 = epsilons5[:1852]

    data6_1 = data6_1[:1819]
    epsilons6 = epsilons6[:1819]

    data7_1 = data7_1[:1334]
    epsilons7 = epsilons7[:1334]

    data8_1 = data8_1[:1001]
    epsilons8 = epsilons8[:1001]

    data9_1 = data9_1[:667]
    epsilons9 = epsilons9[:667]

    data10_1 = data10_1[:572]
    epsilons10 = epsilons10[:572]

    data11_1 = data11_1[:500]
    epsilons11 = epsilons11[:500]

    data12_1 = data12_1[:400]
    epsilons12 = epsilons12[:400]

    data13_1 = data13_1[:334]
    epsilons13 = epsilons13[:334]

    data14_1 = data14_1[:250]
    epsilons14 = epsilons14[:250]

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "24"
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["mathtext.rm"] = "Roman"
    rc = {"font.family": "serif",
          "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    if name_index == 2:
        plt.title(r'Fairness $(\gamma)$ as a function of $\epsilon_{1}$', size=30)
        plt.ylabel('Fairness $(\Gamma)$', size=30)
        plt.xlabel(r'Privacy loss $\epsilon_{1}$', size=30)
        plt.grid()

    if name_index == 4:
        plt.title(r'Accuracy $(\Theta)$ as a function of $\epsilon_{1}$', size=30)
        plt.ylabel(r'Accuracy $(\Theta)$', size=30)
        plt.xlabel(r'Privacy loss $\epsilon_{1}$', size=30)
        plt.grid()

        #fig.suptitle(r'Proportion of True selection in 10,000 runs as a function of $\epsilon_{1}$', size=15)
        #plt.title(r'Accuracy $(\Theta)$ as a function of $\epsilon_{1}$', size=10)
        #plt.ylabel(r'Accuracy $(\Theta)$', size=10)

    if name_index in [0, 1, 3, 5, 6, 7, 8, 9]:
        continue


    xp1 = np.linspace(0, 20, 100000)
    xp2 = np.linspace(0, 20, 100000)
    xp3 = np.linspace(0, 20, 100000)
    xp4 = np.linspace(0, 20, 100000)
    xp5 = np.linspace(0, 20, 100000)
    xp6 = np.linspace(0, 20, 100000)
    xp7 = np.linspace(0, 20, 100000)
    xp8 = np.linspace(0, 20, 100000)
    xp9 = np.linspace(0, 20, 100000)
    xp10 = np.linspace(0, 20, 100000)
    xp11 = np.linspace(0, 20, 100000)
    xp12 = np.linspace(0, 20, 100000)
    xp13 = np.linspace(0, 20, 100000)
    xp14 = np.linspace(0, 20, 100000)
    xp15 = np.linspace(0, 20, 100000)

    z1_1 = np.polyfit(epsilons1, data1_1, 5)
    p1_1 = np.poly1d(z1_1)
    z2_1 = np.polyfit(epsilons2, data2_1, 5)
    p2_1 = np.poly1d(z2_1)
    z3_1 = np.polyfit(epsilons3, data3_1, 5)
    p3_1 = np.poly1d(z3_1)
    z4_1 = np.polyfit(epsilons4, data4_1, 5)
    p4_1 = np.poly1d(z4_1)
    z5_1 = np.polyfit(epsilons5, data5_1, 5)
    p5_1 = np.poly1d(z5_1)
    z6_1 = np.polyfit(epsilons6, data6_1, 5)
    p6_1 = np.poly1d(z6_1)
    z7_1 = np.polyfit(epsilons7, data7_1, 5)
    p7_1 = np.poly1d(z7_1)
    z8_1 = np.polyfit(epsilons8, data8_1, 5)
    p8_1 = np.poly1d(z8_1)
    z9_1 = np.polyfit(epsilons9, data9_1, 5)
    p9_1 = np.poly1d(z9_1)
    z10_1 = np.polyfit(epsilons10, data10_1, 5)
    p10_1 = np.poly1d(z10_1)
    z11_1 = np.polyfit(epsilons11, data11_1, 5)
    p11_1 = np.poly1d(z11_1)
    z12_1 = np.polyfit(epsilons12, data12_1, 5)
    p12_1 = np.poly1d(z12_1)
    z13_1 = np.polyfit(epsilons13, data13_1, 5)
    p13_1 = np.poly1d(z13_1)
    z14_1 = np.polyfit(epsilons14, data14_1, 5)
    p14_1 = np.poly1d(z14_1)
    z15_1 = np.polyfit(epsilons15, data15_1, 5)
    p15_1 = np.poly1d(z15_1)

    plt.plot(xp1, p1_1(xp1), '-', color='cornflowerblue',
             label=r'($\alpha$=$\frac{' + str(proportions[0][0]) + '}{' + str(proportions[0][1]) + '}$)')

    plt.plot(xp2, p2_1(xp2), '-', color='steelblue', label =r'($\alpha$=$\frac{'+str(proportions[1][0])+'}{'+str(proportions[1][1])+'}$)')

    #plt.plot(epsilons3, data3, color='darkgray')
    plt.plot(xp3, p3_1(xp3), '-', color='darkviolet', label =r'($\alpha$=$\frac{'+str(proportions[2][0])+'}{'+str(proportions[2][1])+'}$)')

    #plt.plot(epsilons4, data4, color='darkgray')
    plt.plot(xp4, p4_1(xp4), '-', color='lime', label= r'($\alpha$=$\frac{'+str(proportions[3][0])+'}{'+str(proportions[3][1])+'}$)')
    #plt.plot(epsilons5, data5, color='darkgray')
    plt.plot(xp5, p5_1(xp5), '-', color='blue', label= r'($\alpha$=$\frac{'+str(proportions[4][0])+'}{'+str(proportions[4][1])+'}$)')
    #plt.plot(epsilons6, data6, color='darkgray')
    plt.plot(xp6, p6_1(xp6), '-', color='dodgerblue', label = r'($\alpha$=$\frac{'+str(proportions[5][0])+'}{'+str(proportions[5][1])+'}$)')
    #plt.plot(epsilons7, data7, color='darkgray')
    plt.plot(xp7, p7_1(xp7), '-', color='teal', label = r'($\alpha$=$\frac{'+str(proportions[6][0])+'}{'+str(proportions[6][1])+'}$)')
    #plt.plot(epsilons8, data8, color='darkgray')
    plt.plot(xp8, p8_1(xp8), '-', color='royalblue', label = r'($\alpha$=$\frac{'+str(proportions[7][0])+'}{'+str(proportions[7][1])+'}$)')
    #plt.plot(epsilons9, data9, color='darkgray')
    plt.plot(xp9, p9_1(xp9), '-', color='deepskyblue', label = r'($\alpha$=$\frac{'+str(proportions[8][0])+'}{'+str(proportions[8][1])+'}$)')

    plt.plot(xp10, p10_1(xp10), '-', color='violet',
             label=r'($\alpha$=$\frac{' + str(proportions[9][0]) + '}{' + str(proportions[9][1]) + '}$)')

    plt.plot(xp11, p11_1(xp11), '-', color='red',
             label=r'($\alpha$=$\frac{' + str(proportions[10][0]) + '}{' + str(proportions[10][1]) + '}$)')

    plt.plot(xp12, p12_1(xp12), '-', color='darkorange',
             label=r'($\alpha$=$\frac{' + str(proportions[11][0]) + '}{' + str(proportions[11][1]) + '}$)')

    plt.plot(xp13, p13_1(xp13), '-', color='gold',
             label=r'($\alpha$=$\frac{' + str(proportions[12][0]) + '}{' + str(proportions[12][1]) + '}$)')

    plt.plot(xp14, p14_1(xp14), '-', color='aquamarine',
             label=r'($\alpha$=$\frac{' + str(proportions[13][0]) + '}{' + str(proportions[13][1]) + '}$)')

    plt.plot(xp15, p15_1(xp15), '-', color='darkred',
             label='Debiased model')


    plt.legend(ncol=3, prop={'size':22})
    plt.savefig(str(names[name_index]) + '_single_selection_5.pdf', dpi=300)
    #plt.show()
    #plt.show()
    plt.clf()
    print('hi')


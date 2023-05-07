import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox


epsilons_ratio = [0.01, 0.01, 0.0115, 0.013, 0.0145,
                  0.016, 0.0175, 0.019, 0.021, 0.024]

alpha = ['1_12', '1_1', '115_1', '13_1', '145_1',
         '16_1', '175_1', '19_1', '21_1',
         '24_1']

epsilons1 = [0.01*x for x in range(2000)]
epsilons1[0] = 0.0001

epsilons2 = [0.01*x for x in range(2000)]
epsilons2[0] = 0.0001

epsilons3 = [0.0115*x for x in range(2000)]
epsilons3[0] = 0.0001

epsilons4 = [0.013*x for x in range(2000)]
epsilons4[0] = 0.0001

epsilons5 = [0.0145*x for x in range(2000)]
epsilons5[0] = 0.0001

epsilons6 = [0.016*x for x in range(2000)]
epsilons6[0] = 0.0001

epsilons7 = [0.0175*x for x in range(2000)]
epsilons7[0] = 0.0001

epsilons8 = [0.019*x for x in range(2000)]
epsilons8[0] = 0.0001

epsilons9 = [0.021*x for x in range(2000)]
epsilons9[0] = 0.0001

epsilons10 = [0.024*x for x in range(2000)]
epsilons10[0] = 0.0001

#epsilon10_1 = [0.1*x for x in range(200)]
#epsilon10_2 = [5*x for x in range(200)]
#epsilons10 = epsilon10_1 + epsilon10_2[4:]
epsilons11 = [0.1*x for x in range(200)]
epsilons11[0] = 0.0001

names = ['wha_equal_opportunity', 'wha_accuracy_in_all_iterations']

wha_file_name_1_12 = ['wha_1_12_equal_opportunity', 'wha_1_12_accuracy_in_all_iterations']

same_noise_wha_file_name = ['wha_same_equal_opportunity', 'wha_same_accuracy_in_all_iterations']

wha_file_name_115_1 = ['wha_115_1_equal_opportunity', 'wha_115_1_accuracy_in_all_iterations']

wha_file_name_13_1 = ['wha_13_1_equal_opportunity', 'wha_13_1_accuracy_in_all_iterations']

wha_file_name_145_1 = ['wha_145_1_equal_opportunity', 'wha_145_1_accuracy_in_all_iterations']

wha_file_name_16_1 = ['wha_16_1_equal_opportunity', 'wha_16_1_accuracy_in_all_iterations']

wha_file_name_175_1 = ['wha_175_1_equal_opportunity', 'wha_175_1_accuracy_in_all_iterations']

wha_file_name_19_1 = ['wha_19_1_equal_opportunity', 'wha_19_1_accuracy_in_all_iterations']

wha_file_name_21_1 = ['wha_21_1_equal_opportunity', 'wha_21_1_accuracy_in_all_iterations']

wha_file_name_24_1 = ['wha_24_1_equal_opportunity', 'wha_24_1_accuracy_in_all_iterations']




#adversarial = ['wha_equal_opportunity_debiased_1_4_10000_2Kepsilon_rho_sensitivity_comp_6',
#               'wha_accuracy_in_all_iterations_debiased_1_4_10000_2Kepsilon_rho_sensitivity_comp_6']

adversarial = ['wha_equal_opportunity_debiased_wha_debiased_eo_5_1_200',
               'wha_accuracy_in_all_iterations_debiased_wha_debiased_eo_5_1_200']

expeiments=[same_noise_wha_file_name, wha_file_name_1_12, wha_file_name_115_1, wha_file_name_13_1, wha_file_name_145_1,
              wha_file_name_16_1, wha_file_name_175_1, wha_file_name_19_1, wha_file_name_21_1, wha_file_name_24_1,
              adversarial]
proportions = [['same'], [1, 1.12], [1.15, 1], [1.3, 1], [1.45, 1], [1.6, 1], [1.75, 1], [1.9, 1], [2.1, 1], [2.4, 1]]
for name_index in range(2):
    data1 = pd.read_csv(str(expeiments[0][name_index])+'.csv')
    data2 = pd.read_csv(str(expeiments[1][name_index])+'.csv')
    data3 = pd.read_csv(str(expeiments[2][name_index])+'.csv')
    data4 = pd.read_csv(str(expeiments[3][name_index])+'.csv')
    data5 = pd.read_csv(str(expeiments[4][name_index])+'.csv')
    data6 = pd.read_csv(str(expeiments[5][name_index])+'.csv')
    data7 = pd.read_csv(str(expeiments[6][name_index])+'.csv')
    data8 = pd.read_csv(str(expeiments[7][name_index])+'.csv')
    data9 = pd.read_csv(str(expeiments[8][name_index])+'.csv')
    data10 = pd.read_csv(str(expeiments[9][name_index])+'.csv')
    data11 = pd.read_csv(str(expeiments[10][name_index])+'.csv')

    data1 = data1.drop(columns='Unnamed: 0')
    data2 = data2.drop(columns='Unnamed: 0')
    data3 = data3.drop(columns='Unnamed: 0')
    data4 = data4.drop(columns='Unnamed: 0')
    data5 = data5.drop(columns='Unnamed: 0')
    data6 = data6.drop(columns='Unnamed: 0')
    data7 = data7.drop(columns='Unnamed: 0')
    data8 = data8.drop(columns='Unnamed: 0')
    data9 = data9.drop(columns='Unnamed: 0')
    data10 = data10.drop(columns='Unnamed: 0')
    data11 = data11.drop(columns='Unnamed: 0')
    datas = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11]

    for data_index in range(len(datas)):
        data12 = np.array(datas[data_index])
        datas[data_index] = []
        for each in data12:
            datas[data_index].append(each[0])

    data1 = datas[0]
    data2 = datas[1]
    data3 = datas[2]
    data4 = datas[3]
    data5 = datas[4]
    data6 = datas[5]
    data7 = datas[6]
    data8 = datas[7]
    data9 = datas[8]
    data10 = datas[9]
    data11 = datas[10]

    data3 = data3[:1740]
    epsilons3 = epsilons3[:1740]

    data4 = data4[:1539]
    epsilons4 = epsilons4[:1539]

    data5 = data5[:1380]
    epsilons5 = epsilons5[:1380]

    data6 = data6[:1250]
    epsilons6 = epsilons6[:1250]

    data7 = data7[:1143]
    epsilons7 = epsilons7[:1143]

    data8 = data8[:1053]
    epsilons8 = epsilons8[:1053]

    data9 = data9[:953]
    epsilons9 = epsilons9[:953]

    data10 = data10[:834]
    epsilons10 = epsilons10[:834]


    #data1 = data1[:1000]
    #data2 = data2[:1000]
    #epsilons1 = epsilons1[:1000]
    #fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(9, 8))
    #plt.xticks(fontsize=4)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "22"
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.rcParams["mathtext.rm"] = "Roman"
    rc = {"font.family": "serif",
          "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    if name_index == 0:
        plt.title(r'Fairness $(\gamma)$ as a function of $\epsilon_{1}$', size=28)
        plt.ylabel('Fairness $(\Gamma)$', size=28)
        plt.xlabel(r'Privacy loss $\epsilon_{1}$', size=28)
        plt.grid()

    if name_index == 1:
        #fig.suptitle(r'Proportion of True selection in 10,000 runs as a function of $\epsilon_{1}$', size=15)
        plt.title(r'Accuracy $(\Theta)$ as a function of $\epsilon_{1}$', size=28)
        plt.ylabel(r'Accuracy $(\Theta)$', size=28)
        plt.xlabel(r'Privacy loss $\epsilon_{1}$', size=28)
        plt.grid()


    xp1 = np.linspace(0, 20, 2000)
    xp2 = np.linspace(0, 20, 2000)
    xp3 = np.linspace(0, 20, 2000)
    xp4 = np.linspace(0, 20, 2000)
    xp5 = np.linspace(0, 20, 2000)
    xp6 = np.linspace(0, 20, 2000)
    xp7 = np.linspace(0, 20, 2000)
    xp8 = np.linspace(0, 20, 2000)
    xp9 = np.linspace(0, 20, 2000)
    xp10 = np.linspace(0, 20, 2000)
    xp11 = np.linspace(0, 20, 2000)

    z1 = np.polyfit(epsilons1, data1, 5)
    p1 = np.poly1d(z1)
    z2 = np.polyfit(epsilons2, data2, 5)
    p2 = np.poly1d(z2)
    z3 = np.polyfit(epsilons3, data3, 5)
    p3 = np.poly1d(z3)
    z4 = np.polyfit(epsilons4, data4, 5)
    p4 = np.poly1d(z4)
    z5 = np.polyfit(epsilons5, data5, 5)
    p5 = np.poly1d(z5)
    z6 = np.polyfit(epsilons6, data6, 5)
    p6 = np.poly1d(z6)
    z7 = np.polyfit(epsilons7, data7, 5)
    p7 = np.poly1d(z7)
    z8 = np.polyfit(epsilons8, data8, 5)
    p8 = np.poly1d(z8)
    z9 = np.polyfit(epsilons9, data9, 5)
    p9 = np.poly1d(z9)
    z10 = np.polyfit(epsilons10, data10, 5)
    p10 = np.poly1d(z10)
    z11 = np.polyfit(epsilons11, data11, 7)
    p11 = np.poly1d(z11)

    #plt.plot(epsilons1, data1, color='darkgray')
    plt.plot(xp1, p1(xp1), '-', color='lime', label=r'($\alpha$=1)')
    #plt.plot(epsilons2, data2, color='darkgray')
    plt.plot(xp2, p2(xp2), '-', color='steelblue', label =r'($\alpha$=$\frac{'+str(proportions[1][0])+'}{'+str(proportions[1][1])+'}$)')
    #plt.plot(epsilons3, data3, color='darkgray')
    plt.plot(xp3, p3(xp3), '-', color='darkviolet', label =r'($\alpha$=$\frac{'+str(proportions[2][0])+'}{'+str(proportions[2][1])+'}$)')
    #plt.plot(epsilons4, data4, color='darkgray')
    plt.plot(xp4, p4(xp4), '-', color='cornflowerblue', label= r'($\alpha$=$\frac{'+str(proportions[3][0])+'}{'+str(proportions[3][1])+'}$)')
    #plt.plot(epsilons5, data5, color='darkgray')
    plt.plot(xp5, p5(xp5), '-', color='blue', label= r'($\alpha$=$\frac{'+str(proportions[4][0])+'}{'+str(proportions[4][1])+'}$)')
    #plt.plot(epsilons6, data6, color='darkgray')
    plt.plot(xp6, p6(xp6), '-', color='dodgerblue', label = r'($\alpha$=$\frac{'+str(proportions[5][0])+'}{'+str(proportions[5][1])+'}$)')
    #plt.plot(epsilons7, data7, color='darkgray')
    plt.plot(xp7, p7(xp7), '-', color='teal', label = r'($\alpha$=$\frac{'+str(proportions[6][0])+'}{'+str(proportions[6][1])+'}$)')
    #plt.plot(epsilons8, data8, color='darkgray')
    plt.plot(xp8, p8(xp8), '-', color='royalblue', label = r'($\alpha$=$\frac{'+str(proportions[7][0])+'}{'+str(proportions[7][1])+'}$)')
    #plt.plot(epsilons9, data9, color='darkgray')
    plt.plot(xp9, p9(xp9), '-', color='deepskyblue', label = r'($\alpha$=$\frac{'+str(proportions[8][0])+'}{'+str(proportions[8][1])+'}$)')
    plt.plot(xp10, p10(xp10), '-', color='darkblue', label = r'($\alpha$=$\frac{'+str(proportions[9][0])+'}{'+str(proportions[9][1])+'}$)')
    plt.plot(xp11, p11(xp11), '-', color='darkred', label = r'Debiased model')

    '''
    ax1.plot(epsilons1, data1, color='darkgray')
    ax1.plot(xp1, p1(xp1), '-', color='orangered')
    ax2.plot(epsilons2, data2, color='darkgray')
    ax2.plot(xp2, p2(xp2), '-', color='orangered')
    ax3.plot(epsilons3, data3, color='darkgray')
    ax3.plot(xp3, p3(xp3), '-', color='orangered')
    ax4.plot(epsilons4, data4, color='darkgray')
    ax4.plot(xp4, p4(xp4), '-', color='orangered')
    ax5.plot(epsilons5, data5, color='darkgray')
    ax5.plot(xp5, p5(xp5), '-', color='orangered')
    ax6.plot(epsilons6, data6, color='darkgray')
    ax6.plot(xp6, p6(xp6), '-', color='orangered')
    ax7.plot(epsilons7, data7, color='darkgray')
    ax7.plot(xp7, p7(xp7), '-', color='orangered')
    ax8.plot(epsilons8, data8, color='darkgray')
    ax8.plot(xp8, p8(xp8), '-', color='orangered')
    ax9.plot(epsilons9, data9, color='darkgray')
    ax9.plot(xp9, p9(xp9), '-', color='orangered')
    '''
    legend1 = plt.legend(  # *scatter.legend_elements(num=None),
        # bbox_to_anchor=(0.5, 1.05),
        prop={'size': 12}, ncol=5, fancybox=True, shadow=True, loc="lower right",
        title=r"$\alpha$ values")
    #plt.legend()
    #plt.show()
    #plt.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.savefig(str(names[name_index]) + '_all_in_one_plus_debiased_March.pdf', dpi=300)
    #plt.show()
    plt.clf()
    print('hi')






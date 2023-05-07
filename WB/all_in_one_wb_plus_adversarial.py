import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox


epsilons_ratio = [0.01, 0.013, 0.016, 0.019, 0.022, 0.025, 0.028, 0.031, 0.034, 0.037, 0.04]

alpha = ['1_1', '13_1', '16_1', '19_1', '22_1', '25_1', '28_1', '31_1', '34_1', '37_1', '4_1']


epsilons1 = [0.01*x for x in range(2000)]
epsilons1[0] = 0.0001

epsilons2 = [0.013*x for x in range(2000)]
epsilons2[0] = 0.0001

epsilons3 = [0.016*x for x in range(2000)]
epsilons3[0] = 0.0001

epsilons4 = [0.019*x for x in range(2000)]
epsilons4[0] = 0.0001

epsilons5 = [0.022*x for x in range(2000)]
epsilons5[0] = 0.0001

epsilons6 = [0.025*x for x in range(2000)]
epsilons6[0] = 0.0001

epsilons7 = [0.028*x for x in range(2000)]
epsilons7[0] = 0.0001

epsilons8 = [0.031*x for x in range(2000)]
epsilons8[0] = 0.0001

epsilons9 = [0.034*x for x in range(2000)]
epsilons9[0] = 0.0001

epsilons10 = [0.037*x for x in range(2000)]
epsilons10[0] = 0.0001

epsilons11 = [0.04*x for x in range(2000)]
epsilons11[0] = 0.0001

#epsilon10_1 = [0.1*x for x in range(200)]
#epsilon10_2 = [5*x for x in range(200)]
#epsilons10 = epsilon10_1 + epsilon10_2[4:]
epsilons12 = [0.1 * x for x in range (200)]
epsilons12[0] = 0.0001

names = ['wb_equal_opportunity', 'wb_accuracy_in_all_iterations']

same_noise_wb_file_name = ['wb_same_equal_opportunity', 'wb_same_accuracy_in_all_iterations']

wb_file_name_13_1 = ['wb_13_1_equal_opportunity', 'wb_13_1_accuracy_in_all_iterations']

wb_file_name_16_1 = ['wb_16_1_equal_opportunity', 'wb_16_1_accuracy_in_all_iterations']

wb_file_name_19_1 = ['wb_19_1_equal_opportunity', 'wb_19_1_accuracy_in_all_iterations']

wb_file_name_22_1 = ['wb_22_1_equal_opportunity', 'wb_22_1_accuracy_in_all_iterations']

wb_file_name_25_1 = ['wb_25_1_equal_opportunity', 'wb_25_1_accuracy_in_all_iterations']

wb_file_name_28_1 = ['wb_28_1_equal_opportunity', 'wb_28_1_accuracy_in_all_iterations']

wb_file_name_31_1 = ['wb_31_1_equal_opportunity', 'wb_31_1_accuracy_in_all_iterations']

wb_file_name_34_1 = ['wb_34_1_equal_opportunity', 'wb_34_1_accuracy_in_all_iterations']

wb_file_name_37_1 = ['wb_37_1_equal_opportunity', 'wb_37_1_accuracy_in_all_iterations']

wb_file_name_4_1 = ['wb_4_1_equal_opportunity', 'wb_4_1_accuracy_in_all_iterations']

#adversarial = ['wb_equal_opportunity_debiased_1_4_10000_2Kepsilon_rho_sensitivity_comp_9',
#               'wb_accuracy_in_all_iterations_debiased_1_4_10000_2Kepsilon_rho_sensitivity_comp_9']
#wb_equal_opportunity_debiased_1_4_10000_2Kepsilon_rho_sensitivity_comp_9


#adversarial = ['wb_equal_opportunity_debiased_1',
#               'wb_accuracy_in_all_iterations_debiased_1_4_10000_2Kepsilon_rho_sensitivity_comp_9']

adversarial = ["wb_equal_opportunity_debiased_wb_debiased_eo_3_1_250", "wb_accuracy_in_all_iterations_debiased_wb_debiased_eo_3_1_250"]
expeiments = [same_noise_wb_file_name, wb_file_name_13_1, wb_file_name_16_1,
              wb_file_name_19_1, wb_file_name_22_1, wb_file_name_25_1, wb_file_name_28_1,
              wb_file_name_31_1, wb_file_name_34_1, wb_file_name_37_1, wb_file_name_4_1, adversarial]
proportions = [['same'], [1.3, 1], [1.6, 1],
               [1.9, 1], [2.2, 1], [2.5, 1], [2.8, 1], [3.1, 1], [3.4, 1], [3.7, 1], [4, 1]]
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
    data12 = pd.read_csv(str(expeiments[11][name_index])+'.csv')

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
    data12 = data12.drop(columns='Unnamed: 0')
    datas = [data1, data2, data3,
             data4, data5, data6, data7, data8, data9, data10, data11, data12]

    for data_index in range(len(datas)):
        data13 = np.array(datas[data_index])
        datas[data_index] = []
        for each in data13:
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
    data12 = datas[11]

    #data10 = data10[:200]
    #epsilons10 = epsilons10[:200]

    data2 = data2[:1539]
    epsilons2 = epsilons2[:1539]

    data3 = data3[:1250]
    epsilons3 = epsilons3[:1250]

    data4 = data4[:1053]
    epsilons4 = epsilons4[:1053]

    data5 = data5[:910]
    epsilons5 = epsilons5[:910]

    data6 = data6[:800]
    epsilons6 = epsilons6[:800]

    data7 = data7[:715]
    epsilons7 = epsilons7[:715]

    data8 = data8[:646]
    epsilons8 = epsilons8[:646]

    data9 = data9[:589]
    epsilons9 = epsilons9[:589]

    data10 = data10[:541]
    epsilons10 = epsilons10[:541]

    data11 = data11[:500]
    epsilons11 = epsilons11[:500]

    #data1 = data1[:1000]
    #data2 = data2[:1000]
    #epsilons1 = epsilons1[:1000]
    #fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(9, 8))
    #plt.xticks(fontsize=4)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "22"
    plt.rcParams["figure.figsize"] = (10, 8)
    #plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
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
    xp12 = np.linspace(0, 20, 200)
    #xp10 = np.linspace(0, 20, 1000)

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
    z11 = np.polyfit(epsilons11, data11, 5)
    p11 = np.poly1d(z11)
    z12 = np.polyfit(epsilons12, data12, 5)
    p12 = np.poly1d(z12)
    #x = p12(xp12)
    #a = pd.DataFrame()
    #a['epsilon'] = epsilons12
    #a['accuracy'] = x
    #a.to_csv('debiased_epsilon_accuracy.csv')
    #x[-1] = x[-300]
    #for ch in range (300):
    #    x[-ch] = x[-300]
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
    plt.plot(xp10, p10(xp10), '-', color='mediumslateblue', label = r'($\alpha$=$\frac{'+str(proportions[9][0])+'}{'+str(proportions[9][1])+'}$)')
    plt.plot(xp11, p11(xp11), '-', color='darkblue', label = r'($\alpha$=$\frac{'+str(proportions[10][0])+'}{'+str(proportions[10][1])+'}$)')
    plt.plot(xp12, p12(xp12), '-', color='darkred', label = r'Debiased model')

    '''
    #plt.plot(epsilons1, data1, color='darkgray')
    plt.plot(xp1, p1(xp1), '-', color='lime', label=r'($\frac{\epsilon_{1}}{\epsilon_{2}}$=same)')
    #plt.plot(epsilons2, data2, color='darkgray')
    plt.plot(xp2, p2(xp2), '-', color='steelblue', label =r'($\frac{\epsilon_{1}}{\epsilon_{2}}$=$\frac{'+str(proportions[1][0])+'}{'+str(proportions[1][1])+'}$)')
    #plt.plot(epsilons3, data3, color='darkgray')
    plt.plot(xp3, p3(xp3), '-', color='darkviolet', label =r'($\frac{\epsilon_{1}}{\epsilon_{2}}$=$\frac{'+str(proportions[2][0])+'}{'+str(proportions[2][1])+'}$)')
    #plt.plot(epsilons4, data4, color='darkgray')
    plt.plot(xp4, p4(xp4), '-', color='cornflowerblue', label= r'($\frac{\epsilon_{1}}{\epsilon_{2}}$=$\frac{'+str(proportions[3][0])+'}{'+str(proportions[3][1])+'}$)')
    #plt.plot(epsilons5, data5, color='darkgray')
    plt.plot(xp5, p5(xp5), '-', color='blue', label= r'($\frac{\epsilon_{1}}{\epsilon_{2}}$=$\frac{'+str(proportions[4][0])+'}{'+str(proportions[4][1])+'}$)')
    #plt.plot(epsilons6, data6, color='darkgray')
    plt.plot(xp6, p6(xp6), '-', color='dodgerblue', label = r'($\frac{\epsilon_{1}}{\epsilon_{2}}$=$\frac{'+str(proportions[5][0])+'}{'+str(proportions[5][1])+'}$)')
    #plt.plot(epsilons7, data7, color='darkgray')
    plt.plot(xp7, p7(xp7), '-', color='teal', label = r'($\frac{\epsilon_{1}}{\epsilon_{2}}$=$\frac{'+str(proportions[6][0])+'}{'+str(proportions[6][1])+'}$)')
    #plt.plot(epsilons8, data8, color='darkgray')
    plt.plot(xp8, p8(xp8), '-', color='royalblue', label = r'($\frac{\epsilon_{1}}{\epsilon_{2}}$=$\frac{'+str(proportions[7][0])+'}{'+str(proportions[7][1])+'}$)')
    #plt.plot(epsilons9, data9, color='darkgray')
    plt.plot(xp9, p9(xp9), '-', color='deepskyblue', label = r'($\frac{\epsilon_{1}}{\epsilon_{2}}$=$\frac{'+str(proportions[8][0])+'}{'+str(proportions[8][1])+'}$)')
    '''


    #major_ticks = np.arange(0, 21, 5)
    #minor_ticks = np.arange(0, 21, 1)
    #plt.set_xticks(major_ticks)
    #plt.set_xticks(minor_ticks, minor=True)
    #ax1.grid(which='both')
    #plt.grid(which='minor', alpha=0.2)
    #plt.grid(which='major', alpha=0.5)

    '''
    ax2.set_xticks(major_ticks)
    ax2.set_xticks(minor_ticks, minor=True)
    # ax1.grid(which='both')
    ax2.grid(which='minor', alpha=0.2)
    ax2.grid(which='major', alpha=0.5)

    ax3.set_xticks(major_ticks)
    ax3.set_xticks(minor_ticks, minor=True)
    # ax1.grid(which='both')
    ax3.grid(which='minor', alpha=0.2)
    ax3.grid(which='major', alpha=0.5)

    ax7.set_xticks(major_ticks)
    ax7.set_xticks(minor_ticks, minor=True)
    # ax1.grid(which='both')
    ax7.grid(which='minor', alpha=0.2)
    ax7.grid(which='major', alpha=0.5)

    ax9.set_xticks(major_ticks)
    ax9.set_xticks(minor_ticks, minor=True)
    # ax1.grid(which='both')
    ax9.grid(which='minor', alpha=0.2)
    ax9.grid(which='major', alpha=0.5)

    major_ticks = np.arange(0, 16, 5)
    minor_ticks = np.arange(0, 16, 1)
    ax6.set_xticks(major_ticks)
    ax6.set_xticks(minor_ticks, minor=True)
    # ax1.grid(which='both')
    ax6.grid(which='minor', alpha=0.2)
    ax6.grid(which='major', alpha=0.5)

    ax8.set_xticks(major_ticks)
    ax8.set_xticks(minor_ticks, minor=True)
    # ax1.grid(which='both')
    ax8.grid(which='minor', alpha=0.2)
    ax8.grid(which='major', alpha=0.5)

    major_ticks = np.arange(0, 22, 5)
    minor_ticks = np.arange(0, 22, 1)
    ax4.set_xticks(major_ticks)
    ax4.set_xticks(minor_ticks, minor=True)
    # ax1.grid(which='both')
    ax4.grid(which='minor', alpha=0.2)
    ax4.grid(which='major', alpha=0.5)

    major_ticks = np.arange(0, 23, 5)
    minor_ticks = np.arange(0, 23, 1)
    ax5.set_xticks(major_ticks)
    ax5.set_xticks(minor_ticks, minor=True)
    # ax1.grid(which='both')
    ax5.grid(which='minor', alpha=0.2)
    ax5.grid(which='major', alpha=0.5)
    '''
    #scatter = plt.scatter(x_1D, y_1D, s=50, c=colors, cmap="viridis", alpha=0.8)
    #plt.legend()
    legend1 = plt.legend(#*scatter.legend_elements(num=None),
                        # bbox_to_anchor=(0.5, 1.05),
                        prop={'size': 12}, ncol=5, fancybox=True, shadow=True, loc="lower right",
                        title=r"$\alpha$ values")
    #plt.add_artist(legend1)
    #plt.legend()
    #plt.show()
    #plt.subplots_adjust(hspace=0.5, wspace=0.4)
    #plt.savefig(str(names[name_index]) + '_all_in_one_1_plus_adversarial_1_after_neurips.pdf', dpi=300)
    #plt.savefig(str(names[name_index]) + '_all_in_one_adversarial_March.png', dpi=300)
    #plt.show()
    plt.clf()
    print('hi')


#wb 3-1 royalblue/ 15 epsilon 1
#wb 15-1 dodgerblue/ 15 epsilon 1
#wb 2-1 teal/ 20 epsilon 1
#wb 4-1 deepskyblue/ 20 epsilon 1
#wb 12-1 mediumslateblue/ 12 epsilon 1
#wb 1-1 darkviolet/ 20 epsilon 1
#wb 1-105 steelblue/ 20 epsilon 1
#wb 105-1 cornflowerblue/ 21 epsilon 1
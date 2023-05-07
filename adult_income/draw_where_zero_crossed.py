import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import proplot as pplt

epsilons_ratio = [0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016,
                  0.017, 0.018, 0.019, 0.02, 0.021, 0.022,
                  0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029,
                  0.03, 0.031, 0.032, 0.033,  0.034, 0.035,
                  0.036, 0.037,  0.038,  0.039,  0.04]

alpha = ['1_1', '11_1', '12_1', '13_1',
         '14_1', '15_1', '16_1', '17_1', '18_1',
         '19_1', '2_1', '21_1', '22_1', '23_1',
         '24_1', '25_1', '26_1', '27_1', '28_1',
         '29_1', '3_1',  '31_1',  '32_1',  '33_1',
         '34_1',  '35_1',  '36_1',  '37_1', '38_1',
         '39_1', '4_1']

column_names = ["alpha", "epsilon", "accuracy", "index"]

df = pd.read_csv('adult_3D_2.csv')
x = df["epsilon"]
y = df["accuracy"]
colors = df["alpha"]
#area = (30 * np.random.rand(N))**2  # 0 to 15 point radii


#fig, ax = pplt.subplots()
#ax.scatter(x, y, c=colors, cmap="viridis", alpha=0.5)
#pplt.scatter(x, y, c=colors, cmap="viridis", alpha=0.5)
#viridis_colors = pplt.get_colors('viridis', 27)
alpha_list_int = colors.to_list()
alpha_list = []
for each in alpha_list_int:
    alpha_list.append(str(each))
#fig.legend(alpha_list, viridis_colors, label='alpha values', frame=False, loc='r')
#fig.legend(colors, viridis_colors)
#pplt.show()
x_1D = []
x_list = x.to_list()
for each in x_list:
    x_1D.append(each)
y_1D = []
y_list = y.to_list()
for each in y_list:
    y_1D.append(each)
epsilons15 = [0.1*x for x in range(200)]
epsilons15[0] = 0.0001
#data15_1 = pd.read_csv('adult_wb_accuracy_in_all_iterations_debiased_multiple_selection_1.csv')
data15_1 = pd.read_csv('adult_wb_accuracy_in_all_iterations_debiased_multiple_selection_1_adult_debiased_eo_5_1_5.csv')
data15_1 = data15_1.drop(columns='Unnamed: 0')
data15 = np.array(data15_1)
data15_1 = []
for each in data15:
    data15_1.append(each[0])
xp15 = np.linspace(0, 20, 100000)
z15_1 = np.polyfit(epsilons15, data15_1, 5)
p15_1 = np.poly1d(z15_1)
#plt.plot(xp15, p15_1(xp15), '-', color='darkred', label='Debiased')

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "24"
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["mathtext.rm"] = "Roman"
rc = {"font.family": "serif", "mathtext.fontset": "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

fig, ax = plt.subplots()
ax.set_title(r'Accuracy of perfect fair $(\gamma = 0)$ selection as a function of $\epsilon_{1}, \alpha$', size=20)
ax.set_ylabel('Accuracy $(\Theta)$', size=30)
ax.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=30)
scatter = ax.scatter(x_1D, y_1D, s=50, c=colors, cmap="viridis", alpha=0.8)
ax.plot(xp15, p15_1(xp15), '-', color='darkred', label='Debiased')

legend1 = ax.legend(*scatter.legend_elements(num = None),
                    #bbox_to_anchor=(0.5, 1.05),
                    prop={'size': 18}, ncol=3, fancybox=True, shadow=True, loc="lower right", title=r"$\alpha$ values")
ax.add_artist(legend1)
ax.legend()
plt.savefig('fz_adult_1.pdf')
#plt.show()

print ('hi')
import pandas as pd
import numpy as np
import random
from bisect import bisect_left



number_of_iterations = 10000
number_of_samples = 10


'''
#WHA
alpha_a = 0.9634
Y0a = 0.3099
Y0b = 0.1932
'''

#WB
Y0b = 0.33655060999999405
Y0a = 0.2413262699999973
alpha_a = 0.8793309517363427

P00 = pd.read_csv('P00_a0.csv')
P01 = pd.read_csv('P01_a1.csv')
P11 = pd.read_csv('P11_b1.csv')
P10 = pd.read_csv('P10_b0.csv')

a = [P00, P01, P10, P11]
for each in a:
    each = each.drop('Unnamed: 0', axis=1, inplace=True)

P00 = P00.to_numpy()
P01 = P01.to_numpy()
P10 = P10.to_numpy()
P11 = P11.to_numpy()

CDF = pd.read_csv("CDF.csv")
rho = CDF['Score']/100

epsilons1 = [0.05*x for x in range(2000)]
epsilons1[0] = 0.0001

epsilons2 = [0.01*x for x in range(2000)]
epsilons2[0] = 0.0001

sensitivity = 1

a_times_iterations = []
b_times_iterations = []
a1_times_iterations = []
a0_times_iterations = []
b1_times_iterations = []
b0_times_iterations = []

True_a_iteration = []
True_b_iteration = []
False_a_iteration = []
False_b_iteration = []
for ite in range(number_of_iterations):

    #a_y1_times = []
    #b_y1_times = []
    #a_y0_times = []
    #b_y0_times = []

    Q_a1 = []
    Q_b1 = []
    Q_a0 = []
    Q_b0 = []
    for i in range(number_of_samples):
        randa1 = random.random()
        posa1 = bisect_left(np.cumsum(P01), randa1)
        Q_a1.append(rho[posa1])  # Samples from Pr(R=r|A=0, Y=1)

        randb1 = random.random()
        posb1 = bisect_left(np.cumsum(P11), randb1)
        Q_b1.append(rho[posb1])  # Samples from Pr(R=r|A=1, Y=1)

        randa0 = random.random()
        posa0 = bisect_left(np.cumsum(P00), randa0)
        if posa0 == 198:
            posa0 = 197
        Q_a0.append(rho[posa0])  # Samples from Pr(R=r|A=0, Y=0)

        randb0 = random.random()
        posb0 = bisect_left(np.cumsum(P10), randb0)
        Q_b0.append(rho[posb0])  # Samples from Pr(R=r|A=1, Y=0)

    rand_numa = [random.random() for o in range(number_of_samples)]
    temp0a = []  # Index of A=0 and Y=1
    for j in range(number_of_samples):
        if rand_numa[j] < Y0a:
            temp0a.append(0)
        else:
            temp0a.append(1)
    Q_a_first = [x * y for (x, y) in zip(temp0a, Q_a1)]  # Pr(R=r|A=0, Y=1)
    temp0a_reversed = []  # Index of A=0 and Y=0
    for each in temp0a:
        if each == 0:
            temp0a_reversed.append(1)
        else:
            temp0a_reversed.append(0)
    Q_a_second = [x * y for (x, y) in zip(temp0a_reversed, Q_a0)]  # Pr(R=r|A=0, Y=0)
    Q_a = [x + y for (x, y) in zip(Q_a_first, Q_a_second)]

    rand_numb = [random.random() for o in range(number_of_samples)]
    temp0b = []  # Index of A=1 and Y=1
    for k in range(number_of_samples):
        if rand_numb[k] < Y0b:
            temp0b.append(0)
        else:
            temp0b.append(1)
    Q_b_first = [x * y for (x, y) in zip(temp0b, Q_b1)]  # Pr(R=r|A=1, Y=1)
    temp0b_reversed = []
    for each in temp0b:
        if each == 0:
            temp0b_reversed.append(1)
        else:
            temp0b_reversed.append(0)
    Q_b_second = [x * y for (x, y) in zip(temp0b_reversed, Q_b0)]  # Pr(R=r|A=1, Y=0)
    Q_b = [x + y for (x, y) in zip(Q_b_first, Q_b_second)]

    temp = []  # Index of A=0
    rand_num = [random.random() for l in range(number_of_samples)]
    for t in range(number_of_samples):
        if rand_num[t] < alpha_a:
            temp.append(1)
        else:
            temp.append(0)

    temp_reversed = []  # Index of A=1
    for each in temp:
        if each == 0:
            temp_reversed.append(1)
        else:
            temp_reversed.append(0)

    Qual_first = [x * y for (x, y) in zip(temp, Q_a)]
    Qual_second = [x * y for (x, y) in zip(temp_reversed, Q_b)]
    Qual = [x + y for (x, y) in zip(Qual_first, Qual_second)]

    temp0a_np = np.array(temp0a)
    temp0b_np = np.array(temp0b)
    temp_np = np.array(temp)

    all_from_a = np.where(temp_np == 1)[0]
    all_a_1s = np.where(temp0a_np == 1)[0]
    all_a_0s = np.where(temp0a_np == 0)[0]
    a_y1s = list(set(all_from_a) & set(all_a_1s))
    a_y0s = list(set(all_from_a) & set(all_a_0s))

    all_from_b = np.where(temp_np == 0)[0]
    all_b_1s = np.where(temp0b_np == 1)[0]
    all_b_0s = np.where(temp0b_np == 0)[0]
    b_y1s = list(set(all_from_b) & set(all_b_1s))
    b_y0s = list(set(all_from_b) & set(all_b_0s))

    labels = []
    for ii in range(number_of_samples):
        if ii in b_y1s:
            labels.append(1)
        elif ii in a_y1s:
            labels.append(1)
        else:
            labels.append(0)

    a_times_iterations.append((len(a_y1s)+len(a_y0s)))
    b_times_iterations.append(len(b_y1s)+len(b_y0s))
    a1_times_iterations.append(len(a_y1s))
    b1_times_iterations.append(len(b_y1s))
    a0_times_iterations.append(len(a_y0s))
    b0_times_iterations.append(len(b_y0s))

    True_a_epsilon = []
    True_b_epsilon = []
    False_a_epsilon = []
    False_b_epsilon = []

    for ep in range(len(epsilons1)):
        noise_a = [np.random.laplace(loc=0, scale=sensitivity / epsilons1[ep]) for i in range(number_of_samples)]
        noisy_a = [x * y for (x, y) in zip(temp, noise_a)]
        noise_b = [np.random.laplace(loc=0, scale=sensitivity / epsilons2[ep]) for i in range(number_of_samples)]
        noisy_b = [x * y for (x, y) in zip(temp_reversed, noise_b)]
        noise = [x + y for (x, y) in zip(noisy_a, noisy_b)]
        noisy_samples = [x + y for (x, y) in zip(Qual, noise)]
        max_index = np.argmax(noisy_samples)

        if max_index in a_y1s:
            True_a_epsilon.append(1)
            True_b_epsilon.append(0)
            False_a_epsilon.append(0)
            False_b_epsilon.append(0)
        elif max_index in b_y1s:
            True_a_epsilon.append(0)
            True_b_epsilon.append(1)
            False_a_epsilon.append(0)
            False_b_epsilon.append(0)
        elif max_index in a_y0s:
            True_a_epsilon.append(0)
            True_b_epsilon.append(0)
            False_a_epsilon.append(1)
            False_b_epsilon.append(0)
        elif max_index in b_y0s:
            True_a_epsilon.append(0)
            True_b_epsilon.append(0)
            False_a_epsilon.append(0)
            False_b_epsilon.append(1)

    True_a_iteration.append(True_a_epsilon)
    True_b_iteration.append(True_b_epsilon)
    False_a_iteration.append(False_a_epsilon)
    False_b_iteration.append(False_b_epsilon)

    #print('hi')

print ('hi')
True_a_iteration_df = pd.DataFrame(True_a_iteration)
True_b_iteration_df = pd.DataFrame(True_b_iteration)
False_a_iteration_df = pd.DataFrame(False_a_iteration)
False_b_iteration_df = pd.DataFrame(False_b_iteration)

agg_True_a_iteration_df = True_a_iteration_df.sum(axis=0)
agg_True_b_iteration_df = True_b_iteration_df.sum(axis=0)
agg_False_a_iteration_df = False_a_iteration_df.sum(axis=0)
agg_False_b_iteration_df = False_b_iteration_df.sum(axis=0)

qualified_a_population = sum(a1_times_iterations)
qualified_b_population = sum(b1_times_iterations)

print ('hi')

prob_a = agg_True_a_iteration_df.div(qualified_a_population)
prob_b = agg_True_b_iteration_df.div(qualified_b_population)
equal_opportunity = prob_a - prob_b

accuracy_in_all = (agg_True_a_iteration_df+agg_True_b_iteration_df).div(qualified_a_population+qualified_b_population)
#accuracy_in_a = (agg_True_a_iteration_df).div(qualified_a_population)
#accuracy_in_b = (agg_True_b_iteration_df).div(qualified_b_population)

accuracy_in_all_iterations = (agg_True_a_iteration_df+agg_True_b_iteration_df).div(number_of_iterations)
#accuracy_in_a_iterations = (agg_True_a_iteration_df).div(number_of_iterations)
#accuracy_in_b_iterations = (agg_True_b_iteration_df).div(number_of_iterations)

print ('hi')

#prob_a.to_csv('wb_235_1_prob_a.csv')
#prob_b.to_csv('wb_235_1_prob_b.csv')
equal_opportunity.to_csv('wb_500_1_equal_opportunity.csv')
#accuracy_in_all.to_csv('wb_100_1_accuracy_in_all.csv')
#accuracy_in_a.to_csv('wb_235_1_accuracy_in_a.csv')
#accuracy_in_b.to_csv('wb_235_1_accuracy_in_b.csv')
accuracy_in_all_iterations.to_csv('wb_500_1_accuracy_in_all_iterations.csv')
#accuracy_in_a_iterations.to_csv('wb_235_1_accuracy_in_a_iterations.csv')
#accuracy_in_b_iterations.to_csv('wb_235_1_accuracy_in_b_iterations.csv')

print ('hi')


import pandas as pd
import numpy as np
import random
from bisect import bisect_left

epsilon1_s = [0.0115, 0.012, 0.0125, 0.013, 0.0135, 0.014, 0.0145, 0.0155, 0.016, 0.0165, 0.017, 0.0175, 0.018, 0.0185,
              0.019, 0.0195, 0.0205, 0.021, 0.0215, 0.022, 0.0225, 0.023, 0.0235, 0.024, 0.0245, 0.025, 0.0255, 0.026,
              0.0265, 0.027, 0.0275, 0.028, 0.0285, 0.029, 0.0295]

epsilon2_s = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
              0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

#names = ["same", "102_1", "105_1", "107_1", "108_1", "11_1", "15_1", "2_1", "3_1", "35_1", "4_1", "5_1", "6_1", "8_1",
#         "1_102", "1_105", "1_11"]

names = [ "115_1", "12_1", "125_1", "13_1", "135_1", "14_1", "145_1", "155_1", "16_1", "165_1", "17_1", "175_1", "18_1",
          "185_1", "19_1", "195_1", "205_1", "21_1", "215_1", "22_1", "225_1", "23_1", "235_1", "24_1", "245_1", "25_1",
          "255_1", "26_1", "265_1", "27_1", "275_1", "28_1", "285_1", "29_1", "295_1"]

number_of_iterations = 10000
number_of_samples = 10

df = pd.read_csv('lsat_df.csv')

df = df.drop('Unnamed: 0', axis=1)

df_w = df[df['race'] == 1]  # 18285
df_b = df[df['race'] == 0]  # 1282

df_w_0 = df_w[df_w['label'] == 0]  # 1458
df_w_1 = df_w[df_w['label'] == 1]  # 16827

df_b_0 = df_b[df_b['label'] == 0]  # 490
df_b_1 = df_b[df_b['label'] == 1]  # 792

alpha_a = float(18285) / float(18285 + 1282)
Y0a = float(1458) / float(1458 + 16827)
Y0b = float(490) / float(490 + 792)

for ratio in range(len(names)):
    epsilons1 = [epsilon1_s[ratio]*x for x in range(2000)]
    epsilons1[0] = 0.0001

    epsilons2 = [epsilon2_s[ratio]*x for x in range(2000)]
    epsilons2[0] = 0.0001

    sensitivity = 1

    a_times_iterations = []
    b_times_iterations = []
    a1_times_iterations = []
    a0_times_iterations = []
    b1_times_iterations = []
    b0_times_iterations = []

    True_a_iteration_1 = []
    True_b_iteration_1 = []
    False_a_iteration_1 = []
    False_b_iteration_1 = []

    True_a_iteration_2 = []
    True_b_iteration_2 = []
    False_a_iteration_2 = []
    False_b_iteration_2 = []

    True_a_iteration_3 = []
    True_b_iteration_3 = []
    False_a_iteration_3 = []
    False_b_iteration_3 = []

    True_a_iteration_4 = []
    True_b_iteration_4 = []
    False_a_iteration_4 = []
    False_b_iteration_4 = []

    for ite in range(number_of_iterations):
        Qual = []
        ay1 = []
        ay0 = []
        by1 = []
        by0 = []
        temp = []
        temp_reversed = []
        for i in range(number_of_samples):
            gRand = random.random()
            if gRand > alpha_a:
                temp.append(0)
                temp_reversed.append(1)
                #select from B
                bRand = random.random()
                if bRand > Y0b:
                    # select from B1
                    scoreRand = random.randint(0, 791)
                    data_point = df_b_1.iloc[[scoreRand]]
                    score = data_point['score'].values[0]
                    Qual.append(score)
                    by1.append(i)
                else:
                    # select from B0
                    scoreRand = random.randint(0, 489)
                    data_point = df_b_0.iloc[[scoreRand]]
                    score = data_point['score'].values[0]
                    Qual.append(score)
                    by0.append(i)
            else:
                temp.append(1)
                temp_reversed.append(0)
                #select from A
                aRand = random.random()
                if aRand > Y0a:
                    # select from A1
                    scoreRand = random.randint(0, 16826)
                    data_point = df_w_1.iloc[[scoreRand]]
                    score = data_point['score'].values[0]
                    Qual.append(score)
                    ay1.append(i)
                else:
                    # select from A0
                    scoreRand = random.randint(0, 1457)
                    data_point = df_w_0.iloc[[scoreRand]]
                    score = data_point['score'].values[0]
                    Qual.append(score)
                    ay0.append(i)
        a_times_iterations.append((len(ay1) + len(ay0)))
        b_times_iterations.append(len(by1) + len(by0))
        a1_times_iterations.append(len(ay1))
        b1_times_iterations.append(len(by1))
        a0_times_iterations.append(len(ay0))
        b0_times_iterations.append(len(by0))

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
            #noisy_samples = [x + y for (x, y) in zip(Qual, noise_a)]
            noisy_samples = [x + y for (x, y) in zip(Qual, noise)]
            #max_index = np.argmax(noisy_samples)
            res = sorted(range(len(noisy_samples)), key=lambda sub: noisy_samples[sub])[-4:]
            ta, tb, fa, fb = 0, 0, 0, 0
            True_a_epsilon_m, True_b_epsilon_m, False_a_epsilon_m, False_b_epsilon_m = [], [], [], []
            for max_index in reversed(res):
                if max_index in ay1:
                    ta+=1
                    True_a_epsilon_m.append(ta)
                    True_b_epsilon_m.append(tb)
                    False_a_epsilon_m.append(fa)
                    False_b_epsilon_m.append(fb)
                elif max_index in by1:
                    tb+=1
                    True_a_epsilon_m.append(ta)
                    True_b_epsilon_m.append(tb)
                    False_a_epsilon_m.append(fa)
                    False_b_epsilon_m.append(fb)
                elif max_index in ay0:
                    fa+=1
                    True_a_epsilon_m.append(ta)
                    True_b_epsilon_m.append(tb)
                    False_a_epsilon_m.append(fa)
                    False_b_epsilon_m.append(fb)
                elif max_index in by0:
                    fb+=1
                    True_a_epsilon_m.append(ta)
                    True_b_epsilon_m.append(tb)
                    False_a_epsilon_m.append(fa)
                    False_b_epsilon_m.append(fb)
            True_a_epsilon.append(True_a_epsilon_m)
            True_b_epsilon.append(True_b_epsilon_m)
            False_a_epsilon.append(False_a_epsilon_m)
            False_b_epsilon.append(False_b_epsilon_m)

        True_a_epsilon_df = pd.DataFrame(True_a_epsilon)
        True_b_epsilon_df = pd.DataFrame(True_b_epsilon)
        False_a_epsilon_df = pd.DataFrame(False_a_epsilon)
        False_b_epsilon_df = pd.DataFrame(False_b_epsilon)
        True_a_iteration_1.append(True_a_epsilon_df[0])
        True_b_iteration_1.append(True_b_epsilon_df[0])
        False_a_iteration_1.append(False_a_epsilon_df[0])
        False_b_iteration_1.append(False_b_epsilon_df[0])

        True_a_iteration_2.append(True_a_epsilon_df[1])
        True_b_iteration_2.append(True_b_epsilon_df[1])
        False_a_iteration_2.append(False_a_epsilon_df[1])
        False_b_iteration_2.append(False_b_epsilon_df[1])

        True_a_iteration_3.append(True_a_epsilon_df[2])
        True_b_iteration_3.append(True_b_epsilon_df[2])
        False_a_iteration_3.append(False_a_epsilon_df[2])
        False_b_iteration_3.append(False_b_epsilon_df[2])

        True_a_iteration_4.append(True_a_epsilon_df[3])
        True_b_iteration_4.append(True_b_epsilon_df[3])
        False_a_iteration_4.append(False_a_epsilon_df[3])
        False_b_iteration_4.append(False_b_epsilon_df[3])

        #print('hi')


    qualified_a_population = sum(a1_times_iterations)  # 10207
    qualified_b_population = sum(b1_times_iterations)  # 534

    True_a_iteration_1_df = pd.DataFrame(True_a_iteration_1)
    True_b_iteration_1_df = pd.DataFrame(True_b_iteration_1)
    agg_True_a_iteration_1_df = True_a_iteration_1_df.sum(axis=0)
    agg_True_b_iteration_1_df = True_b_iteration_1_df.sum(axis=0)
    prob_a_1 = agg_True_a_iteration_1_df.div(qualified_a_population)
    prob_b_1 = agg_True_b_iteration_1_df.div(qualified_b_population)
    equal_opportunity_1 = prob_a_1 - prob_b_1
    accuracy_in_all_1 = (agg_True_a_iteration_1_df+agg_True_b_iteration_1_df).div(qualified_a_population+qualified_b_population)
    accuracy_in_all_iterations_1 = (agg_True_a_iteration_1_df+agg_True_b_iteration_1_df).div(number_of_iterations)
    accuracy_in_a_iterations_1 = (agg_True_a_iteration_1_df).div(number_of_iterations)
    accuracy_in_b_iterations_1 = (agg_True_b_iteration_1_df).div(number_of_iterations)

    True_a_iteration_2_df = pd.DataFrame(True_a_iteration_2)
    True_b_iteration_2_df = pd.DataFrame(True_b_iteration_2)
    agg_True_a_iteration_2_df = True_a_iteration_2_df.sum(axis=0)
    agg_True_b_iteration_2_df = True_b_iteration_2_df.sum(axis=0)
    prob_a_2 = agg_True_a_iteration_2_df.div(qualified_a_population)
    prob_b_2 = agg_True_b_iteration_2_df.div(qualified_b_population)
    equal_opportunity_2 = prob_a_2 - prob_b_2
    accuracy_in_all_2 = (agg_True_a_iteration_2_df+agg_True_b_iteration_2_df).div(qualified_a_population+qualified_b_population)
    accuracy_in_all_iterations_2 = (agg_True_a_iteration_2_df+agg_True_b_iteration_2_df).div(number_of_iterations)
    accuracy_in_a_iterations_2 = (agg_True_a_iteration_2_df).div(number_of_iterations)
    accuracy_in_b_iterations_2 = (agg_True_b_iteration_2_df).div(number_of_iterations)

    True_a_iteration_3_df = pd.DataFrame(True_a_iteration_3)
    True_b_iteration_3_df = pd.DataFrame(True_b_iteration_3)
    agg_True_a_iteration_3_df = True_a_iteration_3_df.sum(axis=0)
    agg_True_b_iteration_3_df = True_b_iteration_3_df.sum(axis=0)
    prob_a_3 = agg_True_a_iteration_3_df.div(qualified_a_population)
    prob_b_3 = agg_True_b_iteration_3_df.div(qualified_b_population)
    equal_opportunity_3 = prob_a_3 - prob_b_3
    accuracy_in_all_3 = (agg_True_a_iteration_3_df+agg_True_b_iteration_3_df).div(qualified_a_population+qualified_b_population)
    accuracy_in_all_iterations_3 = (agg_True_a_iteration_3_df+agg_True_b_iteration_3_df).div(number_of_iterations)
    accuracy_in_a_iterations_3 = (agg_True_a_iteration_3_df).div(number_of_iterations)
    accuracy_in_b_iterations_3 = (agg_True_b_iteration_3_df).div(number_of_iterations)

    True_a_iteration_4_df = pd.DataFrame(True_a_iteration_4)
    True_b_iteration_4_df = pd.DataFrame(True_b_iteration_4)
    agg_True_a_iteration_4_df = True_a_iteration_4_df.sum(axis=0)
    agg_True_b_iteration_4_df = True_b_iteration_4_df.sum(axis=0)
    prob_a_4 = agg_True_a_iteration_4_df.div(qualified_a_population)
    prob_b_4 = agg_True_b_iteration_4_df.div(qualified_b_population)
    equal_opportunity_4 = prob_a_4 - prob_b_4
    accuracy_in_all_4 = (agg_True_a_iteration_4_df+agg_True_b_iteration_4_df).div(qualified_a_population+qualified_b_population)
    accuracy_in_all_iterations_4 = (agg_True_a_iteration_4_df+agg_True_b_iteration_4_df).div(number_of_iterations)
    accuracy_in_a_iterations_4 = (agg_True_a_iteration_4_df).div(number_of_iterations)
    accuracy_in_b_iterations_4 = (agg_True_b_iteration_4_df).div(number_of_iterations)

    prob_a_1.to_csv('lsat_wb_'+str(names[ratio])+'_prob_a_1.csv')
    prob_b_1.to_csv('lsat_wb_'+str(names[ratio])+'_prob_b_1.csv')
    equal_opportunity_1.to_csv('lsat_wb_'+str(names[ratio])+'_equal_opportunity_1.csv')
    accuracy_in_all_1.to_csv('lsat_wb_'+str(names[ratio])+'_accuracy_in_all_1.csv')
    accuracy_in_all_iterations_1.to_csv('lsat_wb_'+str(names[ratio])+'_accuracy_in_all_iterations_1.csv')
    accuracy_in_a_iterations_1.to_csv('lsat_wb_'+str(names[ratio])+'_accuracy_in_a_iterations_1.csv')
    accuracy_in_b_iterations_1.to_csv('lsat_wb_'+str(names[ratio])+'_accuracy_in_b_iterations_1.csv')

    prob_a_2.to_csv('lsat_wb_'+str(names[ratio])+'_prob_a_2.csv')
    prob_b_2.to_csv('lsat_wb_'+str(names[ratio])+'_prob_b_2.csv')
    equal_opportunity_2.to_csv('lsat_wb_'+str(names[ratio])+'_equal_opportunity_2.csv')
    accuracy_in_all_2.to_csv('lsat_wb_'+str(names[ratio])+'_accuracy_in_all_2.csv')
    accuracy_in_all_iterations_2.to_csv('lsat_wb_'+str(names[ratio])+'_accuracy_in_all_iterations_2.csv')
    accuracy_in_a_iterations_2.to_csv('lsat_wb_'+str(names[ratio])+'_accuracy_in_a_iterations_2.csv')
    accuracy_in_b_iterations_2.to_csv('lsat_wb_'+str(names[ratio])+'_accuracy_in_b_iterations_2.csv')

    prob_a_3.to_csv('lsat_wb_'+str(names[ratio])+'_prob_a_3.csv')
    prob_b_3.to_csv('lsat_wb_'+str(names[ratio])+'_prob_b_3.csv')
    equal_opportunity_3.to_csv('lsat_wb_'+str(names[ratio])+'_equal_opportunity_3.csv')
    accuracy_in_all_3.to_csv('lsat_wb_'+str(names[ratio])+'_accuracy_in_all_3.csv')
    accuracy_in_all_iterations_3.to_csv('lsat_wb_'+str(names[ratio])+'_accuracy_in_all_iterations_3.csv')
    accuracy_in_a_iterations_3.to_csv('lsat_wb_'+str(names[ratio])+'_accuracy_in_a_iterations_3.csv')
    accuracy_in_b_iterations_3.to_csv('lsat_wb_'+str(names[ratio])+'_accuracy_in_b_iterations_3.csv')

    prob_a_4.to_csv('lsat_wb_'+str(names[ratio])+'_prob_a_4.csv')
    prob_b_4.to_csv('lsat_wb_'+str(names[ratio])+'_prob_b_4.csv')
    equal_opportunity_4.to_csv('lsat_wb_'+str(names[ratio])+'_equal_opportunity_4.csv')
    accuracy_in_all_4.to_csv('lsat_wb_'+str(names[ratio])+'_accuracy_in_all_4.csv')
    accuracy_in_all_iterations_4.to_csv('lsat_wb_'+str(names[ratio])+'_accuracy_in_all_iterations_4.csv')
    accuracy_in_a_iterations_4.to_csv('lsat_wb_'+str(names[ratio])+'_accuracy_in_a_iterations_4.csv')
    accuracy_in_b_iterations_4.to_csv('lsat_wb_'+str(names[ratio])+'_accuracy_in_b_iterations_4.csv')


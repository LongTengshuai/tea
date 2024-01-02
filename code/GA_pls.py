import urllib.request
import pandas as pd
import numpy as np
import random
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression
import seaborn as sns
from matplotlib import pyplot as plt
from deap import base
from deap import creator
from deap import tools

excelFile = pd.read_excel("F:\华南农业大学龙老师组\茶叶\茶叶茶多酚\实验项目\数据\dtest\可见+近红外\SPA\波段\叶绿素a_test.xlsx",
                              header=None)  # 正11_转置
data_array= np.array(excelFile)
y = data_array[:, 0]
x = data_array[:, 1:]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

scaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
scaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
scaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
scaled_y_test = (y_test - y_train.mean()) / y_train.std(ddof=1)
#############################################################################################

creator.create('FitnessMax', base.Fitness, weights=(1.0,))  # for minimization, set weights as (-1.0,)
creator.create('Individual', list, fitness=creator.FitnessMax)
###########################################################################
#设置描述符的探索边界, 因为已经scaled了，所以设置成0,1就行
toolbox = base.Toolbox()
min_boundary = np.zeros(x_train.shape[1])
max_boundary = np.ones(x_train.shape[1]) * 1.0
################################################################################
#选择index
def create_ind_uniform(min_boundary, max_boundary):
    index = []
    for min, max in zip(min_boundary, max_boundary):
        index.append(random.uniform(min, max))
    return index
#individual 个体
#population 种群
toolbox.register('create_ind', create_ind_uniform, min_boundary, max_boundary)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.create_ind)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
#############################################################################################
#对于选出个体进行计算得出评价值
def evalOneMax(individual):
    individual_array = np.array(individual)
    selected_x_variable_numbers = np.where(individual_array > threshold_of_variable_selection)[0]
    selected_scaled_x_train = scaled_x_train[:, selected_x_variable_numbers]
    max_number_of_components = 10
    if len(selected_x_variable_numbers):
        # cross-validation
        pls_components = np.arange(1, min(np.linalg.matrix_rank(selected_scaled_x_train) + 1,
                                          max_number_of_components + 1), 1)
        r2_cv_all = []
        for pls_component in pls_components:
            model_in_cv = PLSRegression(n_components=pls_component)
            estimated_y_train_in_cv = np.ndarray.flatten(
                model_selection.cross_val_predict(model_in_cv, selected_scaled_x_train, scaled_y_train,
                                                  cv=5))
            estimated_y_train_in_cv = estimated_y_train_in_cv * y_train.std(ddof=1) + y_train.mean()
            r2_cv_all.append(1 - sum((y_train - estimated_y_train_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2))
        value = [np.max(r2_cv_all)]
    return value
#######################################################################################
#记入评价函数
toolbox.register('evaluate', evalOneMax)
#加入交叉变换
toolbox.register('mate', tools.cxTwoPoint)
#设置突变几率
toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
#挑选个体
toolbox.register('select', tools.selTournament, tournsize=3)
#种群
random.seed()
pop = toolbox.population(n=len(y_train))
#基础参数
number_of_generation=10
probability_of_crossover = 0.5
probability_of_mutation = 0.2
threshold_of_variable_selection = 0.5
###################################################################################
for generation in range(number_of_generation):
    print('-- Generation {0} --'.format(generation + 1))

    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < probability_of_crossover:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < probability_of_mutation:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    # 选出来的个体(描述符)
    print('  Evaluated %i individuals' % len(invalid_ind))

    pop[:] = offspring
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print('  Min %s' % min(fits))
    print('  Max %s' % max(fits))

best_individual = tools.selBest(pop, 1)[0]
best_individual_array = np.array(best_individual)
selected_x_variable_numbers = np.where(best_individual_array > threshold_of_variable_selection)[0]
########################################################################################
for i in range(67):
    sns.lineplot(data=x[i])
sns.scatterplot(x=selected_x_variable_numbers,y=np.ones(len(selected_x_variable_numbers)))


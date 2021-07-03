import itertools
import random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import tensorflow as tf

from road.road import Road

NUMBER_OF_ITERATIONS = 1000
INITIAL_POPULATION_SIZE = 128
NUMBER_OF_PARENT_TO_SELECT = int(INITIAL_POPULATION_SIZE / 2)


model = Sequential([
    InputLayer(input_shape=(Road.NUMBER_OF_ROWS * Road.NUMBER_OF_ROADS,)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer=Adam(),
    loss=MeanSquaredError
)

initial_weights = model.get_weights()
flatten_weights = [layer_weights.flatten() for layer_weights in initial_weights]
flatten_weights_2 = [list(layer_weights) for layer_weights in flatten_weights]
flatten_weights_3 = list(itertools.chain.from_iterable(flatten_weights_2))

weights_length = len(flatten_weights_3)


def main():
    random.seed(0)
    population = generate_initial_population()
    for iteration in range(NUMBER_OF_ITERATIONS):
        fitness_evaluation = evaluate_fitness(population)
        sort_population_based_on_fitness(population, fitness_evaluation)
        print('Scores of top 5:', [fitness_evaluation[individual] for individual in population[:5]])
        parents = select_fittest_individuals(population)
        children = breed(parents)
        population = replace_least_fit_individuals(population, children)


def generate_initial_population():
    population = []
    for i in range(INITIAL_POPULATION_SIZE):
        population.append(generate_individual())
    return population


def generate_individual():
    weights = [0.0] * weights_length
    for index in range(weights_length):
        weights[index] = random.uniform(-1.0, 1.0)
    return tuple(weights)


def evaluate_fitness(population):
    fitness = dict(
        (individual, evaluate_fitness_of_individual(individual)) for individual in population
    )
    return fitness


def sort_population_based_on_fitness(population, fitness_evaluation):
    population.sort(key=lambda individual: fitness_evaluation[individual], reverse=True)


def evaluate_fitness_of_individual(individual):
    weights = [None] * len(initial_weights)
    a = 0
    for index in range(len(weights)):
        initial_weights_2 = initial_weights[index]
        weights_2 = np.array(individual[a:a + initial_weights_2.size])
        weights_2 = np.resize(weights_2, initial_weights_2.shape)
        weights[index] = weights_2
    model.set_weights(weights)

    random.seed(0)
    road = Road()
    done = False
    total_reward = 0
    road.reset()
    while not done:
        x = np.array([
            list(itertools.chain.from_iterable(road.get_state()))
        ])
        action = tf.math.argmax(model(x)[0]).numpy()
        state, reward, done = road.step(action)
        total_reward += reward

    return total_reward


def select_fittest_individuals(population):
    return population[:NUMBER_OF_PARENT_TO_SELECT]


def breed(parents):
    parent_pairs = []
    parents = parents.copy()
    random.shuffle(parents)
    for index in range(0, len(parents), 2):
        parent_pairs.append((parents[index], parents[index + 1]))

    children = []
    for parent_a, parent_b in parent_pairs:
        children.append((parent_a[0], parent_b[1]))
        children.append((parent_b[0], parent_a[1]))

    return children


def replace_least_fit_individuals(population, children):
    return population[:(len(population) - len(children))] + children


if __name__ == '__main__':
    main()

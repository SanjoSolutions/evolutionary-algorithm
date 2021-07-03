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
INITIAL_POPULATION_SIZE = 18
NUMBER_OF_PARENTS_TO_SELECT = int(INITIAL_POPULATION_SIZE / 2)
if NUMBER_OF_PARENTS_TO_SELECT % 2 == 1:
    NUMBER_OF_PARENTS_TO_SELECT -= 1
MUTATION_CHANCE = 0.02
NUMBER_OF_CHILDREN = 1


model = Sequential([
    InputLayer(input_shape=(Road.NUMBER_OF_ROWS * Road.NUMBER_OF_ROADS + Road.NUMBER_OF_ROWS,)),
    Dense(2, activation='relu'),
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
        i = 20
        print('Iteration ' + str(iteration) + ': Scores of top ' + str(i) + ':', [fitness_evaluation[individual] for individual in population[:i]])
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
        weights[index] = generate_genome()
    return tuple(weights)


def generate_genome():
    return random.uniform(-1.0, 1.0)


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
        a += initial_weights_2.size
    model.set_weights(weights)

    random_state = random.getstate()
    random.seed(0)
    road = Road()
    done = False
    total_reward = 0
    road.reset()
    while not done:
        x = np.array([
            list(1 if cell == Road.OTHER_CAR else 0 for cell in itertools.chain.from_iterable(road.rows)) +
            list(car_index_to_embedding(road.car_index))
        ])
        action = tf.math.argmax(model(x)[0]).numpy()
        state, reward, done = road.step(action)
        total_reward += reward

    random.setstate(random_state)

    return total_reward


def car_index_to_embedding(car_index):
    embedding = [0] * Road.NUMBER_OF_ROADS
    embedding[car_index] = 1
    return tuple(embedding)


def select_fittest_individuals(population):
    return population[:NUMBER_OF_PARENTS_TO_SELECT]


def breed(parents):
    parent_pairs = []
    parents = parents.copy()
    random.shuffle(parents)
    for index in range(0, len(parents), 2):
        parent_pairs.append((parents[index], parents[index + 1]))

    children = []
    for parents in parent_pairs:
        for child_number in range(NUMBER_OF_CHILDREN):
            child = [0.0] * weights_length

            for index in range(len(child)):
                if random.random() <= MUTATION_CHANCE:
                    child[index] = generate_genome()
                else:
                    child[index] = random.choice([parent[index] for parent in parents])

            child = tuple(child)

            children.append(child)

    return children


def replace_least_fit_individuals(population, children):
    return population[:(len(population) - len(children))] + children


if __name__ == '__main__':
    main()

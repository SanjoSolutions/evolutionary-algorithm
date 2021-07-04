import itertools
import random

import MultiNEAT as NEAT
import numpy as np

from road.road import Road
from shared import car_index_to_embedding

params = NEAT.Parameters()

params.PopulationSize = 100

genome = NEAT.Genome(
    0,
    Road.NUMBER_OF_ROWS * Road.NUMBER_OF_ROADS + Road.NUMBER_OF_ROWS + 1,
    0,
    3,
    False,
    NEAT.ActivationFunction.UNSIGNED_SIGMOID,
    NEAT.ActivationFunction.UNSIGNED_SIGMOID,
    0,
    params,
    0,
    0
)

pop = NEAT.Population(genome, params, True, 1.0, 0)


def evaluate(genome):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    random_state = random.getstate()
    random.seed(0)
    road = Road()
    done = False
    total_reward = 0
    road.reset()
    while not done and total_reward < 100:
        x = list(1 if cell == Road.OTHER_CAR else 0 for cell in itertools.chain.from_iterable(road.rows)) + \
            list(car_index_to_embedding(road.car_index)) + \
            [1]
        net.Flush()
        net.Input(x)
        net.Activate()
        output = net.Output()
        action = np.argmax(output)
        state, reward, done = road.step(action)
        total_reward += reward

    random.setstate(random_state)

    return total_reward


def main():
    for generation in range(1000):

        genome_list = NEAT.GetGenomeList(pop)
        fitness_list = NEAT.EvaluateGenomeList_Serial(genome_list, evaluate, display=False)
        NEAT.ZipFitness(genome_list, fitness_list)

        print('Fitness:', max(fitness_list), 'Generation:', generation)

        pop.Epoch()


if __name__ == '__main__':
    main()

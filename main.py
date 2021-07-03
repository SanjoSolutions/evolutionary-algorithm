import random

NUMBER_OF_ITERATIONS = 1000
INITIAL_POPULATION_SIZE = 64
NUMBER_OF_PARENT_TO_SELECT = int(INITIAL_POPULATION_SIZE / 2)


def main():
    population = generate_initial_population()
    print(population)
    for iteration in range(NUMBER_OF_ITERATIONS):
        fitness_evaluation = evaluate_fitness(population)
        parents = select_fittest_individuals(population, fitness_evaluation)
        children = breed(parents)
        population = replace_least_fit_individuals(population, children, fitness_evaluation)
        print(population)


def generate_initial_population():
    population = []
    for i in range(INITIAL_POPULATION_SIZE):
        population.append(generate_individual())
    return population


def generate_individual():
    return (
        random.randrange(0, 10 + 1),
        random.randrange(0, 10 + 1)
    )


def evaluate_fitness(population):
    fitness = dict(
        (individual, evaluate_fitness_of_individual(individual)) for individual in population
    )
    return fitness


def evaluate_fitness_of_individual(individual):
    return 10000 - abs(individual[0] + individual[1] - 10)


def select_fittest_individuals(population, fitness_evaluation):
    sorted_population = list(sorted(population, key=lambda individual: fitness_evaluation[individual], reverse=True))
    return sorted_population[:NUMBER_OF_PARENT_TO_SELECT]


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


def replace_least_fit_individuals(population, children, fitness_evaluation):
    sorted_population = list(sorted(population, key=lambda individual: fitness_evaluation[individual], reverse=True))
    return sorted_population[:(len(sorted_population) - len(children))] + children


if __name__ == '__main__':
    main()

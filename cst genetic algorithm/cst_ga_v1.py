import random
import matplotlib.pyplot as plt
from cst_xfoil_v1 import *

# Parameters
POPULATION_SIZE = 10
MUTATION_RATE = 0.4
GENERATIONS = 20
W_LIM = 0.5

# Fitness function: Maximize the aerodynamic efficiency
def fitness_function(individual):
    wu1, wu2, wu3, wl1, wl2, wl3 = individual
    if wu1>W_LIM or wu2>W_LIM or wu3>W_LIM or wl1<-W_LIM or wl2<-W_LIM or wl3<-W_LIM:
        score = -1e6
    else:
        wu = [wu1, wu2, wu3]
        wl = [wl1, wl2, wl3]
        x_coord, y_coord = cst_airfoil(wl, wu)
        score = cst_score(x_coord, y_coord)
    return score

# Create initial population
def create_population(size):
    return [[round(random.uniform(0, W_LIM),2), round(random.uniform(0, W_LIM),2), round(random.uniform(0, W_LIM),2),\
             round(random.uniform(-W_LIM, 0),2), round(random.uniform(-W_LIM, 0),2), round(random.uniform(-W_LIM, 0),2)] for _ in range(size)]

# Selection: Tournament selection
def select(population):
    tournament = random.sample(population, 2)
    return max(tournament, key=fitness_function)

# Crossover: Single point crossover
def crossover(parent1, parent2):
    # Crossover only the relevant genes (3 genes)
    point = random.randint(1, 2)  # Only crossover between m, p, and t
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutation: Randomly change one of the numbers, ensuring the constraints
def mutate(individual):
    if random.random() < MUTATION_RATE:
        index_to_mutate = random.randint(0, 2)
        if index_to_mutate == 0:  # m
            individual[index_to_mutate] = random.randint(0, 6)
        elif index_to_mutate == 1:  # p
            individual[index_to_mutate] = random.randint(0, 5)
        else:  # t
            individual[index_to_mutate] = random.randint(10, 25)
    return individual

# Main Genetic Algorithm
def genetic_algorithm():
    population = create_population(POPULATION_SIZE)
    best_fitness_over_time = []

    for generation in range(GENERATIONS):
        next_population = []
        
        for _ in range(POPULATION_SIZE // 2):
            parent1 = select(population)
            parent2 = select(population)
            child1, child2 = crossover(parent1, parent2)
            next_population.append(mutate(child1))
            next_population.append(mutate(child2))

        population = next_population

        best_solution = max(population, key=fitness_function)
        best_fitness = fitness_function(best_solution)
        best_fitness_over_time.append(best_fitness)

        print(f'Generation {generation}: Best Solution = {best_solution}, Fitness = {best_fitness}')

    return best_solution, best_fitness_over_time

def vidualize(best):
    wu1, wu2, wu3, wl1, wl2, wl3 = best
    wu = [wu1, wu2, wu3]
    wl = [wl1, wl2, wl3]
    x_coord, y_coord = cst_airfoil(wl, wu)
    polar_data, geometry_data, pressure_data = cst_solver(x_coord, y_coord)
    cst_plotter(polar_data, geometry_data, pressure_data)

if __name__ == "__main__":
    best, fitness_history = genetic_algorithm()
    print(f'Best solution found: {best}, Fitness: {fitness_function(best)}')
    vidualize(best)

    # Visualization
    plt.plot(range(GENERATIONS), fitness_history, marker='o')
    plt.title('Fitness of the Best Individual Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Product of m, p, and t)')
    plt.grid()
    plt.show()

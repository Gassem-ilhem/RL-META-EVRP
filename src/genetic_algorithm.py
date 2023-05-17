import random

class GeneticAlgorithm:
    def __init__(self, pop_size, num_generations, mutation_rate, crossover_rate):
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.parents = []

    def initialize_population(self, solution_length):
        self.population = []
        for i in range(self.pop_size):
            solution = [random.randint(0, 1) for j in range(solution_length)]
            self.population.append(solution)
    # def initialize_population(self, solution_length):
    #     self.population = []
    #     for i in range(self.pop_size):
    #         solution = [random.randint(0, solution_length) for j in range(solution_length)]
    #         self.population.append(solution)
        
        
    def select_parents(self, fitness_scores):
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        parents = []
        for i in range(2):
            parent_index = self.roulette_wheel_selection(probabilities)
            parents.append(self.population[parent_index])
        return parents

    def roulette_wheel_selection(self, probabilities):
        r = random.random()
        index = 0
        while r > 0:
            r -= probabilities[index]
            index += 1
        return index - 1

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(0, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        else:
            return parent1, parent2

    def mutate(self, solution):
        mutated_solution = []
        for bit in solution:
            if random.random() < self.mutation_rate:
                mutated_solution.append(1 - bit)
            else:
                mutated_solution.append(bit)
        return mutated_solution

    def evaluate_fitness(self, fitness_function):
        self.fitness_scores = []
        for solution in self.population:
            fitness = fitness_function(solution)
            self.fitness_scores.append(fitness)

    # def evolve(self, fitness_function):
    #     self.initialize_population(solution_length)
    #     for i in range(self.num_generations):
    #         self.evaluate_fitness(fitness_function)
    #         new_population = []
    #         for j in range(self.pop_size // 2):
    #             parent1, parent2 = self.select_parents(self.fitness_scores)
    #             child1, child2 = self.crossover(parent1, parent2)
    #             child1 = self.mutate(child1)
    #             child2 = self.mutate(child2)
    #             new_population.append(child1)
    #             new_population.append(child2)
    #         self.population = new_population
    #     self.evaluate_fitness(fitness_function)
    #     best_solution_index = self.fitness_scores.index(max(self.fitness_scores))
    #     best_solution = self.population[best_solution_index]
    #     return best_solution
    
    

    # def evolve(self, fitness_function, solution_length):
    #     self.initialize_population(solution_length)
    #     for generation in range(self.num_generations):
    #         population_fitness = [(solution, fitness_function(solution)) for solution in self.population]
    #         population_fitness_sorted = sorted(population_fitness, key=lambda x: x[1])
    #         fittest_solution = population_fitness_sorted[0][0]
    #         fitness_scores = [fitness for _, fitness in population_fitness_sorted]
    #         print(f"Generation {generation+1} - Fittest solution cost: {-population_fitness_sorted[0][1]}")
    #         if -population_fitness_sorted[0][1] == 0:
    #             break
    #         self.select_parents(fitness_scores)
    #         self.perform_crossover()
    #         self.perform_mutation()
    #     return fittest_solution



    def evolve(self, fitness_function, solution_length):
        self.initialize_population(solution_length)
        for generation in range(self.num_generations):
            self.evaluate_fitness(fitness_function)
            new_population = []
            for j in range(self.pop_size // 2):
                parent1, parent2 = self.select_parents(self.fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.append(child1)
                new_population.append(child2)
            self.population = new_population
        self.evaluate_fitness(fitness_function)
        best_solution_index = self.fitness_scores.index(max(self.fitness_scores))
        best_solution = self.population[best_solution_index]
        return best_solution
    
    
    def perform_crossover(self):
        new_population = []
        for i in range(self.pop_size):
            parent_1 = self.parents[random.randint(0, len(self.parents)-1)]
            parent_2 = self.parents[random.randint(0, len(self.parents)-1)]
            child = []
            for gene_index in range(len(parent_1)):
                if random.random() < self.crossover_rate:
                    child.append(parent_1[gene_index])
                else:
                    child.append(parent_2[gene_index])
            new_population.append(child)
        self.population = new_population
    
    
    
    
    # def perform_crossover(self):
    #     new_population = []
    #     for i in range(self.pop_size):
    #         parent_1 = self.parents[random.randint(0, len(self.parents) - 1)]
    #         parent_2 = self.parents[random.randint(0, len(self.parents) - 1)]
    #         child = []
    #         for gene_index in range(len(parent_1)):
    #             if random.random() < self.crossover_rate:
    #                 child.append(parent_1[gene_index])
    #             else:
    #                 child.append(parent_2[gene_index])
    #         new_population.append(child)
    #     self.population = new_population
    
    # def perform_crossover(self):
    #     new_population = []
    #     for i in range(self.pop_size):
    #         parent_1 = self.parents[random.randint(0, len(self.parents) - 1)]
    #         parent_2 = self.parents[random.randint(0, len(self.parents) - 1)]
    #         crossover_point = random.randint(0, len(parent_1) - 1)
    #         child = []
    #         for gene_index in range(crossover_point):
    #             child.append(parent_1[gene_index])
    #         for gene_index in range(crossover_point, len(parent_1)):
    #             child.append(parent_2[gene_index])
    #         new_population.append(child)
    #     self.population = new_population

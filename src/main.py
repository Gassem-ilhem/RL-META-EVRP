from utils import load_instance
from solutions import decode_solution, encode_solution, calculate_route_cost
from genetic_algorithm import GeneticAlgorithm
from dqn_agent import DQNAgent

# Load instance data
instance_data = load_instance("../data/solomon_instances/C101.json")

# Define EVRP problem parameters
num_customers = len(instance_data) - 1
num_vehicles = instance_data["max_vehicle_number"]
vehicle_capacity = instance_data["vehicle_capacity"]
customer_demands = []
customer_locations = []
for i in range(1, num_customers+1):
    if f"customer_{i}" in instance_data:
        customer_demands.append(instance_data[f"customer_{i}"]["demand"])
        customer_locations.append((instance_data[f"customer_{i}"]["coordinates"]["x"], instance_data[f"customer_{i}"]["coordinates"]["y"]))

start_location = (instance_data["depart"]["coordinates"]["x"], instance_data["depart"]["coordinates"]["y"])

# Define fitness function for genetic algorithm
def genetic_algorithm_fitness_function(solution):
    routes = decode_solution(solution, num_customers, num_vehicles)
    total_cost = 0
    for vehicle_index, route in enumerate(routes):
        total_cost += calculate_route_cost(route, start_location, customer_locations, customer_demands)
    return -total_cost

# Initialize genetic algorithm
pop_size = 50
num_generations = 100
mutation_rate = 0.01
crossover_rate = 0.8
genetic_algorithm = GeneticAlgorithm(pop_size, num_generations, mutation_rate, crossover_rate)

# Define solution length
solution_length = num_customers * num_vehicles

# Run genetic algorithm to obtain initial solution
initial_solution = genetic_algorithm.evolve(genetic_algorithm_fitness_function, solution_length)
initial_routes = decode_solution(initial_solution, num_customers, num_vehicles)

# Define functions for DQN agent
def get_state(routes, capacities):
    state = []
    for customer_index in range(num_customers):
        for vehicle_index in range(num_vehicles):
            if customer_index in routes[vehicle_index]:
                state.append(1)
            else:
                state.append(0)
    for capacity in capacities:
        state.append(capacity)
    return state

def perform_action(action, routes, capacities, demands, locations):
    next_routes = [route[:] for route in routes]
    next_capacities = capacities[:]
    if action < num_customers: # Assign customer to vehicle
        customer_index = action
        for vehicle_index, route in enumerate(next_routes):
            if customer_index in route:
                route.remove(customer_index)
                next_capacities[vehicle_index] += demands[customer_index]
        best_vehicle_index = -1
        best_cost = float("inf")
        for vehicle_index, route in enumerate(next_routes):
            if next_capacities[vehicle_index] >= demands[customer_index]:
                candidate_route = route[:]
                candidate_route.append(customer_index)
                candidate_cost = calculate_route_cost(candidate_route, start_location, locations, demands)
                if candidate_cost < best_cost:
                    best_vehicle_index = vehicle_index
                    best_cost = candidate_cost
        if best_vehicle_index >= 0:
            next_routes[best_vehicle_index].append(customer_index)
            next_capacities[best_vehicle_index] -= demands[customer_index]
    else: # Assign vehicle to customer
        vehicle_index = action - num_customers
        best_customer_index = -1
        best_cost = float("inf")
        for customer_index in range(num_customers):
            if customer_index not in next_routes[vehicle_index]:
                candidate_route = next_routes[vehicle_index][:]
                candidate_route.append(customer_index)
                candidate_cost = calculate_route_cost(candidate_route, start_location, locations, demands)
                if candidate_cost < best_cost and next_capacities[vehicle_index] >= demands[customer_index]:
                    best_customer_index = customer_index
                    best_cost = candidate_cost
        if best_customer_index >= 0:
            next_routes[vehicle_index].append(best_customer_index)
            next_capacities[vehicle_index] -= demands[best_customer_index]
    return (next_routes, next_capacities)

def get_reward(routes, capacities, demands, locations):
    total_cost = 0
    for vehicle_index, route in enumerate(routes):
        total_cost += calculate_route_cost(route, start_location, locations, demands)
    return -total_cost

# Initialize DQN agent
state_size = num_customers * num_vehicles + num_vehicles
action_size = num_customers + num_vehicles
dqn_agent = DQNAgent(state_size, action_size)

# Train DQN agent on initial solution
num_episodes = 100
for episode in range(num_episodes):
    state = get_state(initial_routes, [vehicle_capacity] * num_vehicles)
    done = False
    total_reward = 0
    while not done:
        action = dqn_agent.act(state)
        next_routes, next_capacities = perform_action(action, initial_routes, [vehicle_capacity] * num_vehicles, customer_demands, customer_locations)
        next_state = get_state(next_routes, next_capacities)
        reward = get_reward(next_routes, next_capacities, customer_demands, customer_locations)
        total_reward += reward
        dqn_agent.remember(state, action, reward, next_state, done)
        state = next_state
        initial_routes = next_routes
        if len(dqn_agent.memory) > dqn_agent.batch_size:
            dqn_agent.replay()
        if reward == 0:
            done = True
    print(f"Episode {episode+1} - Total reward: {total_reward}")
from utils import load_instance
from solutions import decode_solution, encode_solution, calculate_route_cost
from genetic_algorithm import GeneticAlgorithm
from dqn_agent import DQNAgent
import numpy as np

# Load instance data
instance_data = load_instance('../data/solomon_instances/C101.json')

# Define EVRP problem parameters
num_customers = len(instance_data) - 1
num_vehicles = instance_data["max_vehicle_number"]
vehicle_capacity = instance_data["vehicle_capacity"]
vehicle_chargingRate=instance_data["vehicle_chargingRate"]
vehicle_dischargingRate=instance_data["vehicle_dischargingRate"]
vehicle_initialCharging=instance_data["vehicle_initialCharging"]
vehicle_avgTravelSpeed=instance_data["vehicle_avgTravelSpeed"]
vehicle_battery = np.array([instance_data["vehicle_battery"]] * num_vehicles, dtype=int) 

customer_demands = []
customer_locations = []
for i in range(1, num_customers+1):
    if f"customer_{i}" in instance_data:
        customer_demands.append(instance_data[f"customer_{i}"]["demand"])
        customer_locations.append((instance_data[f"customer_{i}"]["coordinates"]["x"], instance_data[f"customer_{i}"]["coordinates"]["y"]))

start_location = (instance_data["depart"]["coordinates"]["x"], instance_data["depart"]["coordinates"]["y"])

# Define fitness function for genetic algorithm
def genetic_algorithm_fitness_function(solution):
    routes = decode_solution(solution, num_customers, num_vehicles, vehicle_battery, vehicle_initialCharging, vehicle_avgTravelSpeed,customer_locations, vehicle_chargingRate, vehicle_dischargingRate)
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
initial_routes = decode_solution(solution, num_customers, num_vehicles, vehicle_battery, vehicle_initialCharging, vehicle_avgTravelSpeed, vehicle_chargingRate, vehicle_dischargingRate)



# Define functions for DQN agent
def get_state(routes, capacities):
    state = np.zeros((num_vehicles, num_customers + 1))
    for vehicle_index in range(num_vehicles):
        for customer_index in routes[vehicle_index]:
            state[vehicle_index, customer_index] = 1
        state[vehicle_index, num_customers] = capacities[vehicle_index] / vehicle_capacity
    return state.flatten()

def get_reward(routes, capacities, demands, locations):
    total_cost = 0
    for vehicle_index, route in enumerate(routes):
        total_cost += calculate_route_cost(route, start_location, locations, demands)
    return -total_cost


def perform_action(action, routes, capacities, demands, locations):
    next_routes = [route[:] for route in routes]
    next_capacities = [0] * num_vehicles


    if action < num_customers:  # Assign customer to vehicle
        customer_index = action
        for vehicle_index, route in enumerate(next_routes):
            if customer_index in route:
                route.remove(customer_index)
                if customer_index < len(demands):
                    next_capacities[vehicle_index] += demands[customer_index]


        best_vehicle_index = -1
        best_cost = float("inf")
        for vehicle_index, route in enumerate(next_routes):
            for customer_index in range(len(demands)):
                if customer_index < len(demands) and next_capacities[vehicle_index] >= demands[customer_index]:
                    candidate_route = route[:]
                    candidate_route.append(customer_index)
                    candidate_cost = calculate_route_cost(candidate_route, start_location, locations, demands)
                    if candidate_cost < best_cost:
                        best_vehicle_index = vehicle_index
                        best_cost = candidate_cost


        if best_vehicle_index >= 0 and customer_index < len(demands):
            next_routes[best_vehicle_index].append(customer_index)
            next_capacities[best_vehicle_index] -= demands[customer_index]


    else:  # Assign vehicle to customer
        vehicle_index = action - num_customers


        if 0 <= vehicle_index < num_vehicles:
            best_customer_index = -1
            best_cost = float("inf")
            for customer_index in range(num_customers):
                if customer_index not in next_routes[vehicle_index]:
                    candidate_route = next_routes[vehicle_index][:]
                    candidate_route.append(customer_index)
                    candidate_cost = calculate_route_cost(candidate_route, start_location, locations, demands)
                    if candidate_cost < best_cost and customer_index < len(demands) and next_capacities[vehicle_index] >= demands[customer_index]:
                        best_customer_index = customer_index
                        best_cost = candidate_cost


            if best_customer_index >= 0 and best_customer_index < len(demands):
                next_routes[vehicle_index].append(best_customer_index)
                next_capacities[vehicle_index] -= demands[best_customer_index]


    return next_routes, next_capacities


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
    current_routes = initial_routes[:]  # Create a copy of initial_routes
    while not done:
        action = dqn_agent.act(state)
        next_routes, next_capacities = perform_action(action, current_routes, [vehicle_capacity] * num_vehicles, customer_demands, customer_locations)
        next_state = get_state(next_routes, next_capacities)
        reward = get_reward(next_routes, next_capacities, customer_demands, customer_locations)
        total_reward += reward
        dqn_agent.remember(state, action, reward, next_state, done)
        state = next_state
        current_routes = next_routes  # Update current_routes instead of initial_routes
        if len(dqn_agent.memory) > dqn_agent.batch_size:
            dqn_agent.replay()
        if reward == 0:
            done = True
    print(f"Episode {episode+1} - Total reward: {total_reward}")


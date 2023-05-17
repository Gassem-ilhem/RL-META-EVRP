# def decode_solution(solution, num_customers, num_vehicles):
#     routes = [[] for i in range(num_vehicles)]
#     for i, bit in enumerate(solution):
#         vehicle_index = i // num_customers
#         customer_index = i % num_customers
#         if bit == 1:
#             routes[vehicle_index].append(customer_index)
#     return routes
import numpy as np

def decode_solution(solution, num_customers, num_vehicles, vehicle_battery, vehicle_initialCharging, vehicle_avgTravelSpeed, customer_locations, customer_charging_rates, customer_discharging_rates):
    routes = [[] for i in range(num_vehicles)]
    battery_levels = np.zeros(num_vehicles)
    charging_levels = np.zeros(num_vehicles)
    for i, bit in enumerate(solution):
        vehicle_index = i // num_customers
        customer_index = i % num_customers
        if bit == 1:
            # Calculate travel time and charging time for this customer
            current_location = 0 if len(routes[vehicle_index]) == 0 else routes[vehicle_index][-1] + 1
            next_location = customer_index + 1
            distance = np.sqrt((customer_locations[next_location-1][0] - customer_locations[current_location-1][0])**2 + (customer_locations[next_location-1][1] - customer_locations[current_location-1][1])**2)
            travel_time = distance / vehicle_avgTravelSpeed
            charging_time = (vehicle_battery[vehicle_index] - battery_levels[vehicle_index]) / customer_charging_rates[current_location] if battery_levels[vehicle_index] < vehicle_initialCharging else 0

            # Update battery levels at current node
            if current_location > 0:
                battery_levels[vehicle_index] -= travel_time * customer_discharging_rates[current_location]
                battery_levels[vehicle_index] = max(battery_levels[vehicle_index], 0)
                charging_levels[vehicle_index] = 0
            else:
                battery_levels[vehicle_index] = vehicle_initialCharging
                charging_levels[vehicle_index] = battery_levels[vehicle_index]

            # Check if we need to charge at the current node
            if battery_levels[vehicle_index] < vehicle_battery[vehicle_index] * 0.1:
                charging_time = (vehicle_battery[vehicle_index] - battery_levels[vehicle_index]) / customer_charging_rates[current_location]
                battery_levels[vehicle_index] += charging_time * customer_charging_rates[current_location]
                battery_levels[vehicle_index] = min(battery_levels[vehicle_index], vehicle_battery[vehicle_index])
                charging_levels[vehicle_index] = battery_levels[vehicle_index]

            # Update battery levels and charging levels at next node
            if next_location <= num_customers:
                battery_levels[vehicle_index] -= charging_time * customer_discharging_rates[next_location]
                battery_levels[vehicle_index] = max(battery_levels[vehicle_index], 0)
                charging_levels[vehicle_index] = 0

            # Add customer to route
            routes[vehicle_index].append(customer_index)

    return routes


def encode_solution(routes):
    solution = []
    for i in range(len(routes)):
        for j in range(len(routes[i])):
            index = i * len(routes[0]) + routes[i][j]
            solution.append(index)
    return solution


def calculate_route_cost(route, start_location, customer_locations, customer_demands):
    if not route:
        return 0
    distances = []
    current_location = start_location
    for customer_index in route:
        if customer_index < len(customer_locations):
            distances.append(calculate_distance(current_location, customer_locations[customer_index]))
            current_location = customer_locations[customer_index]
        else:
            return float("inf")
    distances.append(calculate_distance(current_location, start_location))
    return sum(distances)
    
    
def calculate_distance(location1, location2):
    x1, y1 = location1
    x2, y2 = location2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def get_state(routes, vehicle_capacities):
    state = []
    for route in routes:
        for customer_index in range(len(vehicle_capacities)):
            if customer_index < len(route):
                state.append(route[customer_index])
            else:
                state.append(-1)
        state.append(vehicle_capacities[vehicle_index])
    return state

def perform_action(action, routes, vehicle_capacities, customer_demands, customer_locations):
    next_routes = [list(route) for route in routes]
    next_vehicle_capacities = list(vehicle_capacities)
    customer_index = action % len(customer_demands)
    vehicle_index = action // len(customer_demands)
    if customer_index in next_routes[vehicle_index]:
        next_routes[vehicle_index].remove(customer_index)
        next_vehicle_capacities[vehicle_index] += customer_demands[customer_index]
        reward = -1
    elif next_vehicle_capacities[vehicle_index] >= customer_demands[customer_index]:
        next_routes[vehicle_index].append(customer_index)
        next_vehicle_capacities[vehicle_index] -= customer_demands[customer_index]
        reward = 1
    else:
        reward = -10
    done = all([len(route) == 0 for route in next_routes])
    next_state = get_state(next_routes, next_vehicle_capacities)
    return next_state, reward, done

def get_routes(state, num_customers, num_vehicles):
    routes = []
    for vehicle_index in range(num_vehicles):
        route = []
        for i in range(num_customers):
            if state[vehicle_index * num_customers + i] != -1:
                route.append(state[vehicle_index * num_customers + i])
        routes.append(route)
    return routes
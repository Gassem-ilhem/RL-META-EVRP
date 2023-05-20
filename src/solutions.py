import numpy as np
import math

import numpy as np

def decode_solution(solution, num_customers, num_vehicles, vehicle_battery, vehicle_initialCharging, vehicle_avgTravelSpeed, customer_locations, vehicle_chargingRate, vehicle_dischargingRate):
    routes = [[] for _ in range(num_vehicles)]
    battery_levels = [0] * num_vehicles  # Initialize battery_levels as a list of zeros
    charging_levels = [0] * num_customers  # Update charging_levels size to num_customers
    vehicle_initialCharging = [100] * num_vehicles

    for i, bit in enumerate(solution):
        vehicle_index = i // num_customers
        customer_index = i % num_customers

        if bit == 1:
            current_location = 0 if len(routes[vehicle_index]) == 0 else routes[vehicle_index][-1] + 1
            next_location = customer_index + 1

            # Calculate travel time and charging time for this customer
            distance = calculate_distance(customer_locations[current_location - 1], customer_locations[next_location - 1])
            travel_time = distance / vehicle_avgTravelSpeed
            charging_time =(vehicle_battery[vehicle_index] - battery_levels[vehicle_index]) / charging_levels[next_location - 1] if battery_levels[vehicle_index] < vehicle_initialCharging[vehicle_index] else 0


            # Update battery levels at current node
            if current_location > 0:
                battery_levels[vehicle_index] -= travel_time * vehicle_dischargingRate[current_location - 1]
                battery_levels[vehicle_index] = max(battery_levels[vehicle_index], 0)
                charging_levels[vehicle_index] = 0
            else:
                battery_levels[vehicle_index] = vehicle_initialCharging[vehicle_index]
                charging_levels[vehicle_index] = battery_levels[vehicle_index]

            # Check if we need to charge at the current node
            if battery_levels[vehicle_index] < vehicle_battery[vehicle_index] * 0.1:
                charging_time = (vehicle_battery[vehicle_index] - battery_levels[vehicle_index]) / charging_levels[current_location - 1]
                battery_levels[vehicle_index] += charging_time * charging_levels[current_location - 1]
                battery_levels[vehicle_index] = min(battery_levels[vehicle_index], vehicle_battery[vehicle_index])
                charging_levels[vehicle_index] = battery_levels[vehicle_index]

            # Update battery levels and charging levels at next node
            if next_location <= num_customers:
                battery_levels[vehicle_index] -= charging_time * vehicle_dischargingRate[next_location - 1]
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

  
    
    
def calculate_route_cost(route, customer_demands, customer_locations, vehicle_initialCharging, vehicle_avgTravelSpeed, vehicle_dischargingRate,vehicle_chargingRate,vehicle_battery):
    total_cost = 0
    current_load = 0
    current_battery = vehicle_initialCharging
    current_location = (0, 0)  # Assuming start location is (0, 0)

    for customer_index in route:
        demand = customer_demands[customer_index]
        distance = calculate_distance(current_location, customer_locations[customer_index])
        travel_time = distance / vehicle_avgTravelSpeed
        current_load += demand
        current_battery -= travel_time * vehicle_dischargingRate
        current_battery = min(current_battery + travel_time * vehicle_chargingRate, vehicle_battery)
        current_location = customer_locations[customer_index]
        total_cost += distance

    distance = calculate_distance(current_location, (0, 0))  # Assuming start location is (0, 0)
    total_cost += distance

    return total_cost

def calculate_travel_time(location1, location2, avg_travel_speed):
    if not isinstance(location1, tuple) or len(location1) != 2:
        raise ValueError("location1 must be a tuple containing two integers")
    x1, y1 = location1
    x2, y2 = location2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    travel_time = distance / avg_travel_speed
    return travel_time


    
    
    
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
        state.append(vehicle_capacities[customer_index])  # Fixed variable name from vehicle_index to i
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
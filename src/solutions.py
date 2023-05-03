def decode_solution(solution, num_customers, num_vehicles):
    routes = [[] for i in range(num_vehicles)]
    for i, bit in enumerate(solution):
        vehicle_index = i // num_customers
        customer_index = i % num_customers
        if bit == 1:
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
import math
import numpy as np
from geopy.distance import geodesic
from sklearn.metrics import pairwise

def getDistance(p1, p2):
    """
    Distance in kilometers between two (latitiude, longitude) points.
    """
    return geodesic(p1, p2).km

def create_data_model(input_locations, vehicle_capacities):
    assert (len(vehicle_capacities) > 0), "Number of vehicles have to be greater than 0"

    """Stores the data_model for the problem."""
    data_model = {}
    data_model['distance_matrix'] = []
    input_locations = [[math.radians(_[0]), math.radians(_[1])] for _ in input_locations]
    data_model['distance_matrix'] = np.ceil(pairwise.haversine_distances(input_locations) * 637100)
    
    num_drops = len(input_locations) - 1
    num_vehicles = len(vehicle_capacities)
    data_model['demands'] = np.ones(num_drops + 1)
    data_model['demands'][0] = 0
    data_model['vehicle_capacities'] = vehicle_capacities
    data_model['num_vehicles'] = num_vehicles
    data_model['depot'] = 0
    data_model['num_drops'] = num_drops

    return data_model

def print_solution(data_model, manager, routing, solution, input_locations):
    total_distance = 0
    total_load = 0
    final_route = {}

    for vehicle_id in range(data_model['num_vehicles']):
        if data_model['num_drops'] <= vehicle_id:
            continue
        index = routing.Start(vehicle_id)
        route_distance = 0
        route_load = 0

        route_locs_x = []
        route_locs_y = []
        route_node_index = []
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)

            route_locs_x.append(input_locations[node_index][0])
            route_locs_y.append(input_locations[node_index][1])
            route_node_index.append(node_index)

            route_load += data_model['demands'][node_index]
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)

        route_locs_x.append(route_locs_x[0])
        route_locs_y.append(route_locs_y[0])
        route_node_index.append(route_node_index[0])

        total_distance += route_distance
        total_load += route_load
        final_route[vehicle_id] = {
            "route_locs_x": route_locs_x,
            "route_locs_y": route_locs_y,
            "route_node_index": route_node_index
        }

    return final_route


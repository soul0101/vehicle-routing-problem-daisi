# vehicle-routing-problem-daisi

One of the most important applications of optimization is vehicle routing, in which the goal is to find the best routes for a fleet of vehicles visiting a set of locations. Usually, "best" means routes with the least total distance or cost. Here are a few examples of routing problems:

1) A package delivery company wants to assign routes for drivers to make deliveries.
2) A cable TV company wants to assign routes for technicians to make residential service calls.
3) A ride-sharing company wants to assign routes for drivers to pick up and drop off passengers.

## Test API Call
``` python
import pydaisi as pyd
vehicle_routing_problem = pyd.Daisi("soul0101/Vehicle Routing Problem ")

input_locations, vehicle_capacities = vehicle_routing_problem.get_dummy_data().value

locations_fig = vehicle_routing_problem.get_locations_plot_plotly(input_locations).value
locations_fig.show()

# Search Parameters
sb_local_mh = "AUTOMATIC"
sb_first_sol = "AUTOMATIC"
carryforward_penalty = 1000
search_timeout = 10

# Run solver
final_route = vehicle_routing_problem.vrp_calculator(input_locations, vehicle_capacities, 
                                                    carryforward_penalty=carryforward_penalty, 
                                                    search_timeout=search_timeout, 
                                                    first_sol_strategy=sb_first_sol, 
                                                    ls_metaheuristic=sb_local_mh).value

# Plot Results
routes_fig = vehicle_routing_problem.get_route_plot_plotly(final_route).value
routes_fig.show()
```

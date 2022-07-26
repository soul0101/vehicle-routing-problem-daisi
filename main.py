import math 
import random
import helper
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

def vrp_calculator(input_locations, vehicle_capacities, carryforward_penalty=1000, search_timeout=10, 
                    first_sol_strategy="AUTOMATIC",
                    ls_metaheuristic="AUTOMATIC"):
    """
    Runs the Optimal Route Calculator

    Parameters
    ----------
    input_locations: 
        A list with its first element as the location array of source and rest elements being location arrays of drops
        Eg: [(lat_source, long_source), (lat_drop1, long_drop1), (lat_drop2, long_drop2), ...]
    vehicle_capacities: 
        A list containing the number of drops each vehicle can visit. (Should have atleast one vehicle)
    carryforward_penalty: 
        Penalty for dropping a delivery location 
    search_timeout: 
        Maximum time to find a solution
    first_sol_strategy: string
        A first solution strategy. Reference: https://developers.google.com/optimization/routing/routing_options#first_sol_options
    ls_metaheuristic: string
        Local Search Option Metaheuristic. Reference: https://developers.google.com/optimization/routing/routing_options#local_search_options

    Returns
    -------
    Dict containing route information for each vehicle:
        {\n
            <vehicle_id>: {\n
                "route_locs_x": List containing latitudes of drops in route (in order)\n
                "route_locs_y" : List containing longitudes of drops in route (in order)\n
                "route_node_index" : List containing drop indexes in route (in order)\n
            }\n
        }\n
    """
    
    #Instantiate the data_model
    data_model = helper.create_data_model(input_locations, vehicle_capacities)


    #Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(data_model['distance_matrix']),
                                           data_model['num_vehicles'], data_model['depot'])


    #Create Routing Model
    routing = pywrapcp.RoutingModel(manager)


    #Create and register a transit callback
    def distance_callback(from_index, to_index):
        """
        Returns the distance between the two nodes
        Convert from routing variable Index to distance matrix NodeIndex.
        """
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data_model['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    #Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


    #Add Capacity constraint
    def demand_callback(from_index):
        """
        Returns the demand of the node
        Convert from routing variable Index to demands NodeIndex
        """
        from_node = manager.IndexToNode(from_index)
        return data_model['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,                                  # Null capacity slack
        data_model['vehicle_capacities'],   # Vehicle maximum capacities
        True,                               # Start cumul to zero
        'Capacity')

    penalty = carryforward_penalty          # Penalty for not delivering to a drop point
    for node in range(1, len(data_model['distance_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    #Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (helper.get_first_sol_strategy(first_sol_strategy))
    search_parameters.local_search_metaheuristic = (helper.get_local_search_metaheuristic(ls_metaheuristic))
    search_parameters.time_limit.FromSeconds(search_timeout)

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        return helper.print_solution(data_model, manager, routing, solution, input_locations)
    else:
        return "No Solution"

def get_route_plot_plotly(final_route, fig=None):
    """
    Returns a Plotly Figure with routes for each vehicle

    Parameters
    ----------
    Dict containing route information for each vehicle:
        {\n
            <vehicle_id>: {\n
                "route_locs_x": List containing latitudes of drops in route (in order)\n
                "route_locs_y" : List containing longitudes of drops in route (in order)\n
                "route_node_index" : List containing drop indexes in route (in order)\n
            }\n
        }\n
    
    Returns
    -------
    fig: plotly.graph_objects.Figure
    """
    color = px.colors.sequential.Inferno
    
    if fig is None:
        fig = go.Figure()

    x_drops = []
    y_drops = []
    x_source = []
    y_source = []

    for vehicle_id, route in final_route.items():
        connector_color = random.choice(color)

        x_drops = np.concatenate((x_drops, route["route_locs_x"][1:]))
        y_drops = np.concatenate((y_drops, route["route_locs_y"][1:]))
        x_source = [route["route_locs_x"][0]]
        y_source = [route["route_locs_y"][0]]

        fig.add_trace(go.Scatter(x=route["route_locs_x"], y=route["route_locs_y"],
                        mode='lines+markers', 
                        name="Vehicle #%s"%(vehicle_id),
                        line_color=connector_color, 
                        marker=dict(opacity=0),
                        legendrank=3,
                        legendgroup='vehicles',
                        hoverinfo="skip"))

    fig.add_trace(go.Scatter(x = x_drops, y = y_drops,
                    mode='markers',
                    name='Drops', 
                    legendgroup='Drops',
                    legendrank=1,
                    marker=dict(color='#848ff0', size=6, 
                    line=dict(width=1,color='DarkSlateGrey'))))

    fig.add_trace(go.Scatter(x = x_source, y = y_source,
                    mode='markers',
                    name='Source',
                    legendrank=2,
                    legendgroup='Source',
                    marker=dict(color='red', size=12, 
                    line=dict(width=1,color='DarkSlateGrey'))))

    fig.update_layout(
        width=700,
        height=500,
        margin=dict(l=50,r=50,b=100,t=100,pad=4),
        title={
        'text': "Vehicle Routing Problem",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },
        xaxis_title="Latitude",
        yaxis_title="Longitude",
        paper_bgcolor="#D3D3D3",
        plot_bgcolor="#C0C0C0",
        font=dict(
            family="monospace",
            size=18,
            color="black"
        )
    )
    return fig

def get_locations_plot_plotly(final_arr, fig = None):
    """
    Returns a Plotly Figure for Source and Drop locations

    Parameters
    ----------
    input_locations: 
        A list with its first element as the location array of source and rest elements being location arrays of drops
        Eg: [(lat_source, long_source), (lat_drop1, long_drop1), (lat_drop2, long_drop2), ...]
    
    Returns
    -------
    fig: plotly.graph_objects.Figure
    """
    if fig is None:
        fig = go.Figure()
        
    fig.add_trace(go.Scatter(x = final_arr[1:, 0], y = final_arr[1:, 1],
                    mode='markers',
                    name='Drops', 
                    marker=dict(color='#848ff0', size=6, 
                    line=dict(width=1,color='DarkSlateGrey'))))

    fig.add_trace(go.Scatter(x = [final_arr[0][0]], y = [final_arr[0][1]],
                    mode='markers',
                    name='Source',
                    marker=dict(color='red', size=12, 
                    line=dict(width=1,color='DarkSlateGrey'))))
    fig.update_layout(
        width=700,
        height=500,
        margin=dict(l=50,r=50,b=100,t=100,pad=4),
        xaxis_title="Latitude",
        yaxis_title="Longitude",
        paper_bgcolor="#D3D3D3",
        plot_bgcolor="#C0C0C0",
        font=dict(
            family="monospace",
            size=18,
            color="black"
        )
    )
    return fig

################################## UI ##############################################

def get_dummy_data():
    """
    Returns dummy data for testing

    Returns
    ----------
    input_locations: 
        A list with its first element as the location array of source and rest elements being location arrays of drops
        Eg: [(lat_source, long_source), (lat_drop1, long_drop1), (lat_drop2, long_drop2), ...]
    vehicle_capacities: 
        A list containing the number of drops each vehicle can visit. (Should have atleast one vehicle)
    """
    df_source = pd.read_csv("./data/source.csv")
    df_drops = pd.read_csv("./data/drops.csv")
    source_location = np.column_stack((df_source['Latitude'], df_source['Longitude']))
    drop_locations = np.column_stack((df_drops['Latitude'], df_drops['Longitude']))
    input_locations = np.concatenate([source_location, drop_locations])

    # Mock vehicle capacities
    num_vehicles = math.ceil((len(input_locations) - 1) / 4)
    vehicle_capacities = 5 * np.ones(num_vehicles)

    return input_locations, vehicle_capacities

def st_ui(final_arr, vehicle_capacities):
    st.write("# Welcome to the Vehicle Route Planner Daisi! 👋")
    st.markdown(
        """
        The goal is to find the best routes for a fleet of vehicles visiting a set of locations. Usually, "best" means routes with the least total distance or cost. Here are a few examples of routing problems:\n
        1) A package delivery company wants to assign routes for drivers to make deliveries.\n
        2) A cable TV company wants to assign routes for technicians to make residential service calls.\n
        3) A ride-sharing company wants to assign routes for drivers to pick up and drop off passengers.\n
        """
    )
    
    locations_fig = get_locations_plot_plotly(final_arr)
    st.plotly_chart(locations_fig)

    st.sidebar.header("Local search options")
    #Local Search Option
    sb_local_mh = st.sidebar.selectbox("Select a local search option", 
                        options=["AUTOMATIC", "GREEDY_DESCENT", "GUIDED_LOCAL_SEARCH", 
                        "SIMULATED_ANNEALING", "TABU_SEARCH"], 
                        help="""
                            AUTOMATIC             - Lets the solver select the metaheuristic.\n
                            GREEDY_DESCENT        - Accepts improving (cost-reducing) local search neighbors until a local minimum is reached.\n
                            GUIDED_LOCAL_SEARCH	  - Uses guided local search to escape local minima (cf. http://en.wikipedia.org/wiki/Guided_Local_Search); this is generally the most efficient metaheuristic for vehicle routing.\n
                            SIMULATED_ANNEALING	  - Uses simulated annealing to escape local minima (cf. http://en.wikipedia.org/wiki/Simulated_annealing).\n
                            TABU_SEARCH	          - Uses tabu search to escape local minima (cf. http://en.wikipedia.org/wiki/Tabu_search).\n
                        """)

    st.sidebar.header("First Solution Strategy")
    #First Solution Strategy
    sb_first_sol= st.sidebar.selectbox("Select a first solution strategy", 
                        options=["AUTOMATIC", "PATH_CHEAPEST_ARC", "PATH_MOST_CONSTRAINED_ARC", 
                        "EVALUATOR_STRATEGY", "SAVINGS", "SWEEP", "CHRISTOFIDES", "ALL_UNPERFORMED",
                        "BEST_INSERTION", "PARALLEL_CHEAPEST_INSERTION", "LOCAL_CHEAPEST_INSERTION",
                        "GLOBAL_CHEAPEST_ARC", "LOCAL_CHEAPEST_ARC", "FIRST_UNBOUND_MIN_VALUE"], 
                        
                        help="""
                            AUTOMATIC - Lets the solver detect which strategy to use according to the model being solved. \n
                            PATH_CHEAPEST_ARC - Starting from a route "start" node, connect it to the node which produces the cheapest route segment, then extend the route by iterating on the last node added to the route.\n
                            PATH_MOST_CONSTRAINED_ARC - Similar to PATH_CHEAPEST_ARC, but arcs are evaluated with a comparison-based selector which will favor the most constrained arc first. To assign a selector to the routing model, use the method ArcIsMoreConstrainedThanArc(). \n
                            EVALUATOR_STRATEGY - Similar to PATH_CHEAPEST_ARC, except that arc costs are evaluated using the function passed to SetFirstSolutionEvaluator(). \n
                            SAVINGS - Savings algorithm (Clarke & Wright).\n
                            SWEEP - Sweep algorithm (Wren & Holliday). \n
                            ALL_UNPERFORMED - Make all nodes inactive. Only finds a solution if nodes are optional (are element of a disjunction constraint with a finite penalty cost).\n
                            BEST_INSERTION - Iteratively build a solution by inserting the cheapest node at its cheapest position; the cost of insertion is based on the global cost function of the routing model. As of 2/2012, only works on models with optional nodes (with finite penalty costs).\n
                            PARALLEL_CHEAPEST_INSERTION - Iteratively build a solution by inserting the cheapest node at its cheapest position; the cost of insertion is based on the arc cost function. Is faster than BEST_INSERTION.\n
                            LOCAL_CHEAPEST_INSERTION - Iteratively build a solution by inserting each node at its cheapest position; the cost of insertion is based on the arc cost function. Differs from PARALLEL_CHEAPEST_INSERTION by the node selected for insertion; here nodes are considered in their order of creation. Is faster than PARALLEL_CHEAPEST_INSERTION.\n
                            GLOBAL_CHEAPEST_ARC - Iteratively connect two nodes which produce the cheapest route segment.\n
                            LOCAL_CHEAPEST_ARC - Select the first node with an unbound successor and connect it to the node which produces the cheapest route segment.\n
                            FIRST_UNBOUND_MIN_VALUE - Select the first node with an unbound successor and connect it to the first available node. This is equivalent to the CHOOSE_FIRST_UNBOUND strategy combined with ASSIGN_MIN_VALUE (cf. constraint_solver.h).\n
                        """)

    st.sidebar.header("Carry-Forward Penalty")
    carryforward_penalty = st.sidebar.number_input("Penalty for dropping a delivery location", value=1000)
    st.sidebar.header("Search Timeout")
    search_timeout = st.sidebar.slider(
            'Select a timeout for searching an optimal solution', 10, 600, 10, step=5, help='Increase the time in-case the solution is not satisfactory')

    generate_route_btn = st.button("Generate Route")

    if generate_route_btn:
        with st.spinner('Finding Optimal Routes...'):
            final_route = vrp_calculator(final_arr, vehicle_capacities, carryforward_penalty=carryforward_penalty, search_timeout=search_timeout, first_sol_strategy=sb_first_sol, ls_metaheuristic=sb_local_mh)
            if type(final_route) is not str:
                routes_fig = get_route_plot_plotly(final_route)
                st.header("Generated Routes")
                st.plotly_chart(routes_fig)
                st.json(final_route, expanded=False)
            else:
                st.write("Failed to find a solution, try changing the search parameters or increase the search timeout!")

if __name__ == '__main__':
    input_locations, vehicle_capacities = get_dummy_data()
    st_ui(input_locations, vehicle_capacities)

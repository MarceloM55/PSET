import json
import numpy as np
from scipy.stats import norm

def generate_mean_arrival():
    # Generate mean arrival time randomly within a specified range
    mean_arrival = np.random.uniform(50, 70)  # Example range: 50 to 70
    mean_departure = mean_arrival + 16 * (0.5 + np.random.uniform(0, 1))  # Example range: 50 to 70
    std_deviation = np.random.uniform(2, 8)  # Example range: 2 to 8
    return mean_arrival, mean_departure, std_deviation

def generate_scenario(scenario_id, num_subscenarios=5, std_arrival=15, mean_departure=60, std_departure=15,
                      mean_soc_ini=0.5, std_soc_ini=0.2, mean_arrival_list=[], mean_departure_list=[], std_deviation_list=[]):
    scenario = {str(scenario_id): {}}
    mean_arrival, mean_departure, std_deviation_arrival = generate_mean_arrival()
    mean_arrival_list.append(mean_arrival)
    mean_departure_list.append(mean_departure)
    std_deviation_list.append(std_deviation_arrival)
    std_deviation_departure = std_departure
    probabilities = [0.023, 0.136, 0.682, 0.136, 0.023]

    for i in range(1, num_subscenarios + 1):
        if i == 1:
            departure_time = int(mean_departure - 2 * std_deviation_departure)
            arrival_time = int(mean_arrival - 2 * std_deviation_arrival)
        elif i == 2:
            departure_time = int(mean_departure - std_deviation_departure)
            arrival_time = int(mean_arrival - std_deviation_arrival)
        elif i == 3:
            departure_time = int(mean_departure)
            arrival_time = int(mean_arrival)
        elif i == 4:
            departure_time = int(mean_departure + std_deviation_departure)
            arrival_time = int(mean_arrival + std_deviation_arrival)
        elif i == 5:
            departure_time = int(mean_departure + 2 * std_deviation_departure)
            arrival_time = int(mean_arrival + 2 * std_deviation_arrival)

        soc_ini = np.clip(np.random.normal(mean_soc_ini, std_soc_ini), 0, 1)

        # Determine probability based on the specified distribution
        probability = probabilities[i - 1]

        scenario[str(scenario_id)][str(i)] = {
            "arrival": [arrival_time],
            "departure": [departure_time],
            "SoCini": [soc_ini],
            "probability": probability
        }

    return scenario, mean_arrival_list, mean_departure_list, std_deviation_list

def generate_multiple_scenarios():
    num_scenarios = np.random.randint(1, 11)  # Randomly determine the number of scenarios
    scenarios = []
    mean_arrival_list = []
    mean_departure_list = []
    std_deviation_list = []
    for i in range(1, num_scenarios + 1):
        scenario, mean_arrival_list, mean_departure_list, std_deviation_list = generate_scenario(i, mean_arrival_list=mean_arrival_list,
                                                                                                   mean_departure_list=mean_departure_list,
                                                                                                   std_deviation_list=std_deviation_list)
        scenarios.append(scenario)

    return scenarios, mean_arrival_list, mean_departure_list, std_deviation_list

# Generate scenarios
scenarios, mean_arrival_list, mean_departure_list, std_deviation_list = generate_multiple_scenarios()

# Export to JSON file
with open("RANDOMEV.json", "w") as file:
    json.dump(scenarios, file, indent=4)

print("Scenarios exported to RANDOMEV.json")

# Print lists of mean arrival and departure times, and standard deviations
print("Mean Arrival Times:", mean_arrival_list)
print("Mean Departure Times:", mean_departure_list)
print("Standard Deviations:", std_deviation_list)

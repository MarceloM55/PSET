import json
import random

def generate_scenario():
    scenario = {
        "mean_arrival": [],
        "mean_departure": [],
        "Emax": [],
        "SoCini": [],
        "Probability": []
    }
    return scenario

def generate_subsubscenario(subscenario, subsubscenario_num):
    subsubscenario = {
        "arrival": [subscenario["arrival"][subsubscenario_num - 1]],
        "departure": [subscenario["departure"][subsubscenario_num - 1]]
    }
    return subsubscenario

def generate_subscenario(original_scenario, subscenario_num, std_deviation):
    subscenario = {
        "arrival": [],
        "departure": [],
        "Emax": original_scenario["Emax"],
        "SoCini": original_scenario["SoCini"]
    }
    
    num_subscenarios = 5
    mean_diff = (original_scenario["mean_arrival"][0] - original_scenario["mean_departure"][0]) / (num_subscenarios + 1)
    step = random.uniform(mean_diff - std_deviation, mean_diff + std_deviation)
    
    for i in range(1, num_subscenarios + 1):
        arrival_time = round(original_scenario["mean_arrival"][0] - i * step)
        departure_time = round(original_scenario["mean_departure"][0] - i * step)
        subscenario["arrival"].append(arrival_time)
        subscenario["departure"].append(departure_time)
    
    subscenario["arrival"].sort()
    subscenario["departure"].sort()
    
    return subscenario

def generate_subscenarios(original_scenarios, std_deviation):
    subscenarios = {}
    for key, original_scenario in original_scenarios.items():
        # Randomly choose number of subscenarios to join (between 1 and 5)
        num_to_join = random.randint(1, 5)
        subscenario_keys = random.sample([f"{key}_sub{i}" for i in range(1, 6)], min(num_to_join, 5))
        
        # Create joined scenario
        joined_scenario = generate_scenario()
        for subscenario_key in subscenario_keys:
            if subscenario_key in original_scenarios:
                subscenario = original_scenarios[subscenario_key]
                joined_scenario["Emax"].extend(subscenario["Emax"])
                joined_scenario["SoCini"].extend(subscenario["SoCini"])
                
                # Calculate mean arrival and departure times for the third subsubscenario
                subsubscenario_num = 3
                mean_arrival = subscenario["arrival"][subsubscenario_num - 1]
                mean_departure = subscenario["departure"][subsubscenario_num - 1]
                
                # Append mean arrival and departure times to the joined scenario
                joined_scenario["mean_arrival"].append(mean_arrival)
                joined_scenario["mean_departure"].append(mean_departure)
        
        # Add the fixed probability values
        joined_scenario["Probability"] = [0.023, 0.136, 0.682, 0.136, 0.023]
        
        subscenarios[key] = joined_scenario
        
        # Include original subsubscenarios
        for subscenario_key in subscenario_keys:
            if subscenario_key in original_scenarios:
                subscenarios[subscenario_key] = original_scenarios[subscenario_key]
        
        # Generate subscenarios
        for subsubscenario_key, subsubscenario_num in zip(subscenario_keys, range(1, num_to_join+1)):
            if subsubscenario_key in original_scenarios:
                subsubscenario = generate_subscenario(original_scenario, subsubscenario_num, std_deviation)
                subscenarios[subsubscenario_key] = subsubscenario
                # Generate subsubscenarios
                for subsubsubscenario_num in range(1, 6):  # Generate 5 subsubscenarios
                    subsubsubscenario_key = f"{subsubscenario_key}_sub{subsubsubscenario_num}"
                    subsubsubscenario = generate_subsubscenario(subsubscenario, subsubsubscenario_num)
                    subscenarios[subsubsubscenario_key] = subsubsubscenario
    
    return subscenarios

def main():
    num_scenarios = 1  # Generate only 1 scenario
    std_deviation = 10  # Change this to set the standard deviation for spacing
    original_scenarios = {str(i): generate_scenario() for i in range(num_scenarios)}
    subscenarios = generate_subscenarios(original_scenarios, std_deviation)
    subscenarios_json = json.dumps(subscenarios, indent=4)
    print(subscenarios_json)

if __name__ == "__main__":
    main()

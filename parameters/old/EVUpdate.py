import json
import random

def generate_random_scenario():
    scenario = {}
    num_ev = random.randint(1, 7)  # Random number of EVs between 1 and 7
    scenario["arrival"] = [0] * num_ev
    scenario["departure"] = [0] * num_ev
    scenario["Emax"] = [0] * num_ev
    scenario["SoCini"] = [0] * num_ev

    # Generate arrival and departure times
    for i in range(num_ev):
        if i == 0:
            scenario["arrival"][i] = random.randint(1, 86)
        else:
            min_arrival = scenario["departure"][i-1] + 1
            max_arrival = min(96, scenario["departure"][i-1] + 10)
            if min_arrival <= max_arrival:
                scenario["arrival"][i] = random.randint(min_arrival, max_arrival)
            else:
                return None  # Skip scenario if constraints cannot be met

        min_departure = scenario["arrival"][i] + 8  # Ensure departure[n] >= arrival[n] + 8
        max_departure = min(96, scenario["arrival"][i] + 10)
        if min_departure <= max_departure:
            scenario["departure"][i] = random.randint(min_departure, max_departure)
        else:
            return None  # Skip scenario if constraints cannot be met

        # Generate Emax between 30 and 50
        scenario["Emax"][i] = random.randint(30, 50)
        # Generate SoCini between 0 and 0.5, rounded to three decimal places
        scenario["SoCini"][i] = round(random.uniform(0, 0.5), 3)

    return scenario

scenarios = {}
while len(scenarios) < 10:
    scenario = generate_random_scenario()
    if scenario is not None:
        scenarios[str(len(scenarios) + 1)] = scenario

# Export scenarios to JSON file
with open("EVscenarios3.json", "w") as json_file:
    json.dump(scenarios, json_file, indent=4)

print("Scenarios exported to EVscenarios.json")

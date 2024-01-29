import random
import json

def generate_scenario(num_scenarios):
    scenarios = {}

    for i in range(1, num_scenarios + 1):
        valid_scenario = False

        while not valid_scenario:
            n_cars = int(random.random()*10) + 1
            arrival = sorted([random.randint(1, 96) for _ in range(n_cars)])
            departure = sorted([random.randint(arrival[j], 96) for j in range(n_cars)])
            EVmax = [random.randint(20, 50) for _ in range(n_cars)]
            SoCini = [random.uniform(0, 1) for _ in range(n_cars)]

            # Check if d1 - a1 > 16, d2 - a2 > 16, and d3 - a3 > 16
            # Check if a1 < d1 < a2 < d2 < a3 < d3
            valid = 1
            for k in range(n_cars - 1):
                if not(departure[k] - arrival[k] > 16 and arrival[k] < departure[k] < arrival[k+1]):
                    valid = 0
            if valid == 1:
                valid_scenario = True

        scenarios[str(i)] = {
            "arrival": arrival,
            "departure": departure,
            "EVmax": EVmax,
            "SoCini": SoCini
        }

    return scenarios

num_scenarios = 20  # You can change this number to generate more scenarios
result = generate_scenario(num_scenarios)

# Export to JSON file
with open('RANDOMEV_filtered.json', 'w') as json_file:
    json.dump(result, json_file, indent=2)

print("Filtered scenarios exported to RANDOMEV_filtered.json")

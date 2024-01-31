import random
import json

def generate_scenario(num_scenarios):
    scenarios = {}

    for i in range(1, num_scenarios + 1):
        valid_scenario = False

        while not valid_scenario:
            arrival = sorted([random.randint(1, 96) for _ in range(3)])
            departure = sorted([random.randint(arrival[j], 96) for j in range(3)])
            EVmax = [random.randint(20, 50) for _ in range(3)]
            SoCini = [random.uniform(0, 1) for _ in range(3)]

            # Check if d1 - a1 > 16, d2 - a2 > 16, and d3 - a3 > 16
            # Check if a1 < d1 < a2 < d2 < a3 < d3
            if (departure[0] - arrival[0] > 16 and departure[1] - arrival[1] > 16 and departure[2] - arrival[2] > 16
                    and arrival[0] < departure[0] < arrival[1] < departure[1] < arrival[2] < departure[2]):
                valid_scenario = True

        scenarios[str(i)] = {
            "arrival": arrival,
            "departure": departure,
            "EVmax": EVmax,
            "SoCini": SoCini
        }

    return scenarios

num_scenarios = 10  # You can change this number to generate more scenarios
result = generate_scenario(num_scenarios)

# Export to JSON file
with open('RANDOMEV_filtered.json', 'w') as json_file:
    json.dump(result, json_file, indent=2)

print("Filtered scenarios exported to RANDOMEV_filtered.json")
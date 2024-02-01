import random
import json

def generate_data():
    n = random.randint(2, 6)  # Randomize the number of sets of data between 2 and 6
    std_deviation = random.randint(2, 6)  # Generate a random standard deviation between 2 and 6
    data = {}
    probabilities = [0.023, 0.136, 0.682, 0.136, 0.023]
    
    for i in range(n):
        mean_arrival = sorted([random.randint(1, 96) for _ in range(n)])  # Sort mean arrival times
        mean_departure = [random.randint(1, 96) for _ in range(n)]
        Emax = [random.randint(20, 50) for _ in range(n)]
        SoCini = [random.random() for _ in range(n)]
        
        subscenarios = {}
        for j in range(n):
            mak = mean_arrival[j]
            mdk = mean_departure[j]
            
            subscenarios[f"sub{j+1}"] = {
                "arrival": [mak],
                "departure": [mdk],
                f"sub{j+1}sub1": {
                    "arrival": [mak - 2 * std_deviation],
                    "departure": [mdk - 2 * std_deviation]
                },
                f"sub{j+1}sub2": {
                    "arrival": [mak - std_deviation],
                    "departure": [mdk - std_deviation]
                },
                f"sub{j+1}sub3": {
                    "arrival": [mak],
                    "departure": [mdk]
                },
                f"sub{j+1}sub4": {
                    "arrival": [mak + std_deviation],
                    "departure": [mdk + std_deviation]
                },
                f"sub{j+1}sub5": {
                    "arrival": [mak + 2 * std_deviation],
                    "departure": [mdk + 2 * std_deviation]
                }
            }
        
        data[str(i)] = {
            "mean_arrival": mean_arrival,
            "mean_departure": mean_departure,
            "Emax": Emax,
            "SoCini": SoCini,
            "Probability": probabilities
        }
        data[str(i)].update(subscenarios)
    
    return data

generated_data = generate_data()
print(json.dumps(generated_data, indent=4))



# Export data to a file
with open("EvNormal.json", "w") as json_file:
    json.dump(generated_data, json_file, indent=4)

print("Data exported to EvNormal.json")
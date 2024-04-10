import matplotlib.pyplot as plt
import numpy as np

# Sample data (a single vector)
vector = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    )

# Assuming time intervals are evenly spaced and starting from 00:00
time_in_minutes = np.arange(len(vector))
hours = time_in_minutes // 4  # Assuming each data point represents 15 minutes
minutes = (time_in_minutes % 4) * 15
formatted_time = [f"{h:02d}:{m:02d}" for h, m in zip(hours, minutes)]

time_value_pairs = list(zip(formatted_time, vector))

# Print the time and value side by side
print("Time   Value")
for time, value in time_value_pairs:
    print(f"{time:<7} {value}")
# Plotting
plt.plot(formatted_time, vector, marker='', linestyle='-')

# Adding labels and title
plt.xlabel('Time')
plt.ylabel('Energy Cost (USD/kWh)')
plt.title('Energy Cost vs Time')

# Displaying only times spaced 2 hours apart
plt.xticks(np.arange(0, len(formatted_time), 8), formatted_time[::8], rotation=45)

# Removing the grid
plt.grid(False)

# Displaying the plot
plt.grid(True)
plt.show()

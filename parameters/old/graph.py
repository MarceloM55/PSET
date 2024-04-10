import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Given data
data = [
    "370085,4229000000\t752063,1602",
    "370307,1739\t700000",
    "370916,2462\t650000",
    "375051,201\t600000",
    "386640,3597\t550000",
    "408241,2379\t500000",
    "449446,1085\t450000",
    "495316,7102\t400000",
    "541730,8236\t350000",
    "681317,214\t300000",
    "1640086,457\t250000",
    "3694247,66\t200000"
]

# Initialize empty lists for x and y coordinates
x_coords = []
y_coords = []
annotated_points = [9] 
# Iterate over each line of data
for line in data:
    # Split the line based on the tab character
    parts = line.split('\t')
    
    # Split each part based on the comma to separate x and y values
    x_str, y_str = parts[0].split(','), parts[1].split(',')
    
    # Convert strings to integers and handle the decimal point for x
    x = int(x_str[0]) + int(x_str[1]) / 10**len(x_str[1])
    
    # Convert strings to integers and handle the decimal point for y
    if len(y_str) > 1:
        y = int(y_str[0]) + int(y_str[1]) / 10**len(y_str[1])
    else:
        y = int(y_str[0])
    
    # Append x and y coordinates to their respective lists
    x_coords.append(x)
    y_coords.append(y)

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x_coords, y_coords, color='blue')

# Set titles for x and y axes with custom font
font = FontProperties()
font.set_family('serif')  # Choose your desired font family
font.set_name('Times New Roman')  # Specify the font name
font.set_size(12)  # Set the font size


plt.annotate(f'Over 47% Carbon Reduction, with 34% cost increase ', (x_coords[7], y_coords[7]), textcoords="offset points", xytext=(20, 20), arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
plt.annotate(f'Over 60% Carbon Reduction, with 84% cost increase ', (x_coords[9], y_coords[9]), textcoords="offset points", xytext=(20, 20), arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))


# Set titles for x and y axes
plt.xlabel('Total cost [Hundred Thousand USD]', fontproperties=font)
plt.ylabel('Carbon Emissions over period [kg]', fontproperties=font)

# Set title for the graph
plt.title('Cost and carbon analysis over 12 year period', fontproperties=font)

# Display the plot

plt.show()

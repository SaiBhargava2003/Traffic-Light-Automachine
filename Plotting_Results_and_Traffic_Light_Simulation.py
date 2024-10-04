import matplotlib.pyplot as plt

# Flatten predicted and actual traffic data
predicted_traffic = predictions.numpy().flatten()
actual_traffic = y_test.flatten()

# Plot the results
plt.plot(predicted_traffic, label='Predicted Traffic')
plt.plot(actual_traffic, label='Actual Traffic')
plt.legend()
plt.title('Traffic Prediction vs Actual')
plt.show()

# Function to simulate traffic light durations
def simulate_traffic_lights(predicted_traffic):
    green_light_time = max(5, min(60, 60 * predicted_traffic))
    red_light_time = 60 - green_light_time
    return green_light_time, red_light_time

# Simulate traffic lights for the first 10 predictions
for i in range(10):
    green, red = simulate_traffic_lights(predicted_traffic[i])
    print(f'Green Light Time: {green:.2f} sec, Red Light Time: {red:.2f} sec')

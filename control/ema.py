def exponential_moving_average(data, alpha, previous_ema):
    ema = alpha * data + (1 - alpha) * previous_ema
    return ema

# Example usage:
alpha = 0.2
previous_ema = 0.0

# Simulating detected results (replace this with your actual detection process)
detected_results = [50, 120, 80, 200, 90, 150, 120, 180]

for data_point in range(0, 641, 100):
    previous_ema = exponential_moving_average(data_point, alpha, previous_ema)
    print(f"Data Point: {data_point}, EMA: {previous_ema}")

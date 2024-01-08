def smooth(object_cx, filter_val, smoothed_val):
    if filter_val > 1.0:
        filter_val = 0.99
    elif filter_val <= 0.0:
        filter_val = 0.0

    smoothed_val = (object_cx * (1 - filter_val)) + (smoothed_val * filter_val)

    return int(smoothed_val)

# Example usage and testing:
filter_val = 0.001
smoothed_val = 0.0

for object_cx_value in range(0, 641, 100):
    smoothed_val = smooth(object_cx_value, filter_val, smoothed_val)
    print(f"Input: {object_cx_value}, Smoothed Value: {smoothed_val}")

def map_value(x, in_min, in_max, out_min, out_max):
    # Ensure x is within the input range
    x = max(in_min, min(x, in_max))

    # Perform the mapping
    mapped_value = (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min

    return mapped_value

# Test with values ranging from -480 to 480 in steps of 100
for x in range(-480, 481, 100):
    mapped_output = map_value(x, -480, 480, -60, 60)
    print(f"For input {x}, mapped output is {mapped_output}")

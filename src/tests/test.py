def custom_sort(dataset, i, j):
    # This creates a key that looks like:
    # (row[i], row[j], row[j-1], row[j-2], ..., row[0])
    
    dataset.sort(key=lambda row: (
        row[i], 
        row[j], 
        *[row[k] for k in range(j - 1, -1, -1)]
    ))
    return dataset

data = [
    [1, 10, 5, 2],
    [0, 10, 5, 2],
    [1, 5,  5, 2],
    [1, 10, 5, 1]
]

# Sort by index 3, then 1, then 0
# i = 3, j = 1
sorted_data = custom_sort(data, 3, 1)
print(sorted_data)
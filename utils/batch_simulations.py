import os
def Batch_Simulations(rain_ids, batch_size, destination):
    # Initialize batch count and list to hold current batch of keys
    batch_count = 1
    current_batch = []

    if not os.path.exists(destination):
        os.makedirs(destination)


    # Loop through keys and write to batch files
    for key in rain_ids:
        current_batch.append(str(key))
        
        # If current batch reaches the batch_size, write to file and reset
        if len(current_batch) == batch_size:
            with open(os.path.join(destination, f'batch{batch_count}.txt'), 'w') as f:
                f.write('\n'.join(current_batch))
            
            # Reset for next batch
            current_batch = []
            batch_count += 1

    # Handle any remaining keys
    if current_batch:
        with open(os.path.join(destination, f'batch{batch_count}.txt'), 'w') as f:
            f.write('\n'.join(current_batch))
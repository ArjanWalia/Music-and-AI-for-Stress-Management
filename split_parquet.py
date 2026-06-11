import pyarrow.parquet as pq
import pyarrow as pa
import os

input_file = 'outputs/qiriro_features-001.parquet'
output_dir = 'outputs'

# Open the massive parquet file as a stream
parquet_file = pq.ParquetFile(input_file)

# We will group the internal row groups into 4 separate files
num_groups = parquet_file.num_row_groups
groups_per_file = max(1, num_groups // 4)

print(f"Total row groups to process: {num_groups}")

current_group_data = []
file_count = 1

for i in range(num_groups):
    # Read just one small row group into memory at a time
    row_group = parquet_file.read_row_group(i)
    current_group_data.append(row_group)
    
    # When we have gathered enough groups, write them to a new smaller file
    if len(current_group_data) == groups_per_file or i == num_groups - 1:
        # Combine the chunks into a single table
        sub_table = pa.concat_tables(current_group_data)
        
        output_file = os.path.join(output_dir, f'qiriro_features-001_part{file_count}.parquet')
        pq.write_table(sub_table, output_file)
        
        print(f"Successfully created: {output_file}")
        
        # Clear memory for the next batch
        current_group_data = []
        file_count += 1

print("Finished splitting successfully!")

from data_generation import gen_data
"""
Execute this file to create 100 datasets via random seeds.
"""
data_dir = "./data/sim_data"

for seed in range(1, 100):
    d = gen_data(seed)
    d.to_csv(f"sim_data_seed={seed}.csv", index=False)

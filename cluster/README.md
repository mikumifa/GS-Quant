## Data Preparing & Precompute:

To enable precompute, you need to put a file named "openai_api.key" (with your OpenAI API key in there) under ```code/precompute```, then run the following command with a specified dataset (FB16K-237 in this case):
```bash
cd cluster
python cluster.py --dataset FB15K-237 --output_dir ../../processed_data  # precomputation for seed hierarchy
cd llm_refine
python llm_refine.py --dataset FB15K-237  --model gpt-4o-2024-05-13 # LLM-Guided Hierarchy Refinement (LHR)
cd ..
python cluster.py --dataset FB15K-237  --output_dir ../../processed_data # precomputation for llm hierarchy
```

where the first call of ```cluster.py``` is used to build seed hierarchy; ```llm_refine.py``` is used to refine the seed hierarchy with LLM; The second call of ```cluster.py``` is used to build the final hierarchy with LLM.



python cluster/cluster.py --dataset PrimeKG  --output_dir processed_data
python cluster/cluster.py --dataset YAGO3-10  --output_dir processed_data
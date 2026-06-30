# HeteroMRTA
Code for RAL Paper: Heterogeneous Multi-robot Task Allocation and Scheduling via Reinforcement Learning.

This is a repository using deep reinforcement learning to address single-task agent (ST) multi-robot task(MR) task assignment problem.
We train agents make decisions sequentially, and then they are able to choose task in a decentralized manner in execution.

## Demo

<img src="env/demo.gif" alt="demo" style="width: 70%;">

## Code structure

Three main structures of the code are as below:
1. Environments: generate random tasks locations/ requirements and agents with their depot.
1. Neural network: network based on attention in Pytorch 
1. Ray framework: REINFORCE algorithm implementation in ray.

## Running instructions
1. Set hyperparameters in parameters.py then run ```python driver.py```
2. Testing the trained model by running ```python test.py```

1. requirements: 
    1. python => 3.6
    1. torch >= 1.8.1
    1. numpy, ray, matplotlib, scipy, pandas

## OR-Tools baseline for RALTestSets

The `RALTestSets/` directory is local data and is intentionally ignored by git. To run the pkl-only RAL test sets with OR-Tools:

```powershell
conda create -n mrta-ortools python=3.12 pip -y
conda activate mrta-ortools
pip install numpy==1.26.4 pandas pyyaml natsort matplotlib ortools==9.10.4067

cd baseline
python ORTools.py --dataset RALTestSet_M2_1 --limit 2
python ORTools.py --dataset RALTestSet_M2_1
python ORTools.py --all
```

This runner writes `env_i/ortools.solution` and `<dataset>/ortools.csv`. It is an OR-Tools replacement route generator for pkl-only data, not the original Gurobi CTAS-D or TACO solver.

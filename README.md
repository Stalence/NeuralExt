# Neural Extensions
This is the official repo for the paper  "Neural Set Function Extensions: Learning with Discrete Functions in High Dimensions", presented at Neurips 2022. 

![alt text](https://github.com/Stalence/NeuralExt/blob/main/extensions.jpg)



1. You may create a conda environment using the environment.yml file.
2. To run our code, have a look at the scripts directory. To run a given script you can just use bash, e.g. to use the Lovasz Extension for maximum clique:
```
bash scripts\testing\ENZYMES\run_lovasz.sh
```
More examples:
- Neural extension built on top of the Lovasz Extension
 ```
bash scripts\testing\ENZYMES\ENZYMES_clique\run_neural_lovasz_v4.sh
```
- K-cardinality Lovasz extension:
 ```
bash scripts\testing\ENZYMES\ENZYMES_k_clique\run_lovasz_fixed_k_k4.sh
```

The core logic of the implementation can be found in the "model.py" file inside the "model" folder. 
The extension implementations can be found in the extensions folder.
For the baselines you will find an implementation of the Erdos baseline and the Reinforcement learning baseline in the "models" folder. There are corresponding scripts again in this case, e.g.
 ```
bash scripts\testing\ENZYMES\ENZYMES_clique\reinforce.sh
```

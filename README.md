# federated-reinforcement-learning-cartpole
An attempt to simulate collaboration between quadrupedal robots while keeping the environment shared and local.

Setup :

- It is strongly advised that you install anaconda or any other virtual environment manager to run this project. Dependency resolution is a very sticky spot with this codebase.

- Activate your virtual environment and run -
    
    ```pip install -r requirements.txt```


Execution instructions :

1. For the singular cartpole environment implementing the markov chains that determine the state - 

    ```cd cartpole-plain```

    ```python cartpole_ql.py```

2. For the federated system of cartpoles - 

    ```cd federatedRLCartpole/rl/rl_main```

    ```python main.py```


### References
- https://github.com/glenn89/FederatedRL
- https://github.com/aikorea/awesome-rl
- https://github.com/nishantkr18/federated-model-averaging-for-DQN
- https://github.com/openai/gym
- https://github.com/openai/baselines 
- https://pytorch.org/

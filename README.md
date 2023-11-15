# PPO-Discrete-Pytorch
This is a **clean and robust Pytorch implementation of PPO on Discrete action space**. Here is the result:  

<img src="https://github.com/XinJingHao/DQN-DDQN-Pytorch/blob/main/IMGs/Render_CVP1.gif" width="90%" height="auto">  | <img src="https://github.com/XinJingHao/DQN-DDQN-Pytorch/blob/main/IMGs/Render%20of%20DDQN.gif" width="90%" height="auto">
:-----------------------:|:-----------------------:|

<img src="https://github.com/XinJingHao/PPO-Discrete-Pytorch/blob/main/result.jpg" width="95%" height="auto">

All the experiments are trained with same hyperparameters. **Other RL algorithms by Pytorch can be found [here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch).**

## Dependencies
```python
gymnasium==0.29.1  
box2d-py==2.3.5 
numpy==1.26.1
pytorch==2.1.0
tensorboard==2.15.1

python==3.11.5
```

## How to use my code
### Train from scratch

```bash
python main.py
```
where the default enviroment is CartPole-v1.  

### Play with trained model
```bash
python main.py --write False --render True --Loadmodel True --ModelIdex 300000
```


### Change Enviroment
If you want to train on different enviroments
```bash
python main.py --EnvIdex 1
```
The --EnvIdex can be set to be 0 and 1, where   
```bash
'--EnvIdex 0' for 'CartPole-v1'  
'--EnvIdex 1' for 'LunarLander-v2'   
```
if you want train on LunarLander-v2, you need to install [box2d-py](https://gymnasium.farama.org/environments/box2d/) first. 


### Visualize the training curve
You can use the [tensorboard](https://pytorch.org/docs/stable/tensorboard.html) to visualize the training curve. History training curve is saved at '\runs'

### Hyperparameter Setting
For more details of Hyperparameter Setting, please check 'main.py'


### References
[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)  
[Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf)

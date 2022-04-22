# PPO-Discrete-Pytorch
This is a **clean and robust Pytorch implementation of PPO on Discrete action space**. Here is the result:  
  
![avatar](https://github.com/XinJingHao/PPO-Discrete-Pytorch/blob/main/result.jpg)  
All the experiments are trained with same hyperparameters. **Other RL algorithms by Pytorch can be found [here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch).**

## Dependencies
gym==0.18.3  
numpy==1.21.2  
pytorch==1.8.1  
tensorboard==2.5.0 

## How to use my code
### Train from scratch
run **'python main.py'**, where the default enviroment is CartPole-v1.  
### Play with trained model
run **'python main.py --write False --render True --Loadmodel True --ModelIdex 300000'**  
### Change Enviroment
If you want to train on different enviroments, just run **'python main.py --EnvIdex 1'**.  
The --EnvIdex can be set to be 0~1, where   
'--EnvIdex 0' for 'CartPole-v1'  
'--EnvIdex 1' for 'LunarLander-v2'   
### Visualize the training curve
You can use the tensorboard to visualize the training curve. History training curve is saved at '\runs'
### Hyperparameter Setting
For more details of Hyperparameter Setting, please check 'main.py'
### References
[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)  
[Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf)

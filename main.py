import gym
import torch
import numpy as np
from PPO import device, PPO_discrete
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse



def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=300000, help='which model to load')

parser.add_argument('--seed', type=int, default=209, help='random seed')
parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
parser.add_argument('--Max_train_steps', type=int, default=5e7, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=64, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--l2_reg', type=float, default=0, help='L2 regulization coefficient for Critic')
parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory')
parser.add_argument('--entropy_coef', type=float, default=0, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
parser.add_argument('--adv_normalization', type=str2bool, default=False, help='Advantage normalization')
opt = parser.parse_args()
print(opt)


def evaluate_policy(env, model, render):
    scores = 0
    turns = 3
    for j in range(turns):
        s, done, ep_r, steps = env.reset(), False, 0, 0
        while not done:
            # Take deterministic actions at test time
            a, pi_a = model.evaluate(torch.from_numpy(s).float().to(device))
            s_prime, r, done, info = env.step(a)

            ep_r += r
            steps += 1
            s = s_prime
            if render:
                env.render()
        scores += ep_r
    return scores/turns

def main():
    EnvName = ['CartPole-v1','LunarLander-v2']
    BrifEnvName = ['CP-v1','LLd-v2']
    env_with_Dead = [True, True]
    EnvIdex = opt.EnvIdex
    env = gym.make(EnvName[EnvIdex])
    eval_env = gym.make(EnvName[EnvIdex])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_e_steps = env._max_episode_steps

    write = opt.write
    if write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BrifEnvName[EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    T_horizon = opt.T_horizon
    render = opt.render
    Loadmodel = opt.Loadmodel
    ModelIdex = opt.ModelIdex #which model to load
    Max_train_steps = opt.Max_train_steps #in steps
    eval_interval = opt.eval_interval #in steps
    save_interval = opt.save_interval #in steps

    seed = opt.seed
    torch.manual_seed(seed)
    env.seed(seed)
    eval_env.seed(seed)
    np.random.seed(seed)

    print('Env:',BrifEnvName[EnvIdex],'  state_dim:',state_dim,'  action_dim:',action_dim,'   Random Seed:',seed, '  max_e_steps:',max_e_steps)
    print('\n')

    kwargs = {
        "env_with_Dead": env_with_Dead[EnvIdex],
        "state_dim": state_dim,
        "action_dim": action_dim,
        "gamma": opt.gamma,
        "lambd": opt.lambd,
        "net_width": opt.net_width,
        "lr": opt.lr,
        "clip_rate": opt.clip_rate,
        "K_epochs": opt.K_epochs,
        "batch_size": opt.batch_size,
        "l2_reg":opt.l2_reg,
        "entropy_coef":opt.entropy_coef,  #hard env needs large value
        "adv_normalization":opt.adv_normalization,
        "entropy_coef_decay": opt.entropy_coef_decay,
    }

    if not os.path.exists('model'): os.mkdir('model')
    model = PPO_discrete(**kwargs)
    if Loadmodel: model.load(ModelIdex)


    traj_lenth = 0
    total_steps = 0
    while total_steps < Max_train_steps:
        s, done, steps, ep_r = env.reset(), False, 0, 0

        '''Interact & trian'''
        while not done:
            traj_lenth += 1
            steps += 1
            if render:
                # a, pi_a = model.select_action(torch.from_numpy(s).float().to(device))  #stochastic policy
                a, pi_a = model.evaluate(torch.from_numpy(s).float().to(device))  #deterministic policy
                env.render()
            else:
                a, pi_a = model.select_action(torch.from_numpy(s).float().to(device))

            s_prime, r, done, info = env.step(a)

            if (done and steps != max_e_steps):
                if EnvIdex == 1:
                    if r <=-100: r = -30  #good for LunarLander
                dw = True  #dw: dead and win
            else:
                dw = False

            model.put_data((s, a, r, s_prime, pi_a, done, dw))
            s = s_prime
            ep_r += r

            '''update if its time'''
            if not render:
                if traj_lenth % T_horizon == 0:
                    a_loss, c_loss, entropy = model.train()
                    traj_lenth = 0
                    if write:
                        writer.add_scalar('a_loss', a_loss, global_step=total_steps)
                        writer.add_scalar('c_loss', c_loss, global_step=total_steps)
                        writer.add_scalar('entropy', entropy, global_step=total_steps)

            '''record & log'''
            if total_steps % eval_interval == 0:
                score = evaluate_policy(eval_env, model, False)
                if write:
                    writer.add_scalar('ep_r', score, global_step=total_steps)
                print('EnvName:',BrifEnvName[EnvIdex],'seed:',seed,'steps: {}k'.format(int(total_steps/1000)),'score:', score)
            total_steps += 1

            '''save model'''
            if total_steps % save_interval==0:
                model.save(total_steps)

    env.close()

if __name__ == '__main__':
    main()

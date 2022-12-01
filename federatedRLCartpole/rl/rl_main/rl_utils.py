import json
import sys

import paho.mqtt.client as mqtt
from torch import optim
from environments.gym.frozenlake import FrozenLake_v0
sys.path.insert(0, '/home/sr42/Projects/federated-reinforcement-learning-cartpole/federatedRLCartpole/rl/rl_main')
from main_constants import *

from environments.gym.breakout import BreakoutDeterministic_v4
from environments.gym.cartpole import CartPole_v0, CartPole_v1
from environments.gym.pendulum import Pendulum_v0
from environments.gym.gridworld import GRIDWORLD_v0
from environments.gym.blackjack import Blackjack_v0
from environments.real_device.environment_rip import EnvironmentRIP
#from environments.unity.chaser_unity import Chaser_v1
#from environments.unity.drone_racing import Drone_Racing
from environments.mujoco.inverted_double_pendulum import InvertedDoublePendulum_v2
from environments.mujoco.hopper import Hopper_v2
from environments.mujoco.ant import Ant_v2
from environments.mujoco.half_cheetah import HalfCheetah_v2
from environments.mujoco.swimmer import Swimmer_v2
from environments.mujoco.reacher import Reacher_v2
from environments.mujoco.humanoid import Humanoid_v2
from environments.mujoco.humanoid_stand_up import HumanoidStandUp_v2
from environments.mujoco.inverted_pendulum import InvertedPendulum_v2
from environments.mujoco.walker_2d import Walker2D_v2
from models.actor_critic_model import ActorCriticModel
#from algorithms_rl.DQN_v0 import DQN_v0
#from algorithms_rl.Monte_Carlo_Control_v0 import Monte_Carlo_Control_v0

sys.path.insert(0, '/home/sr42/Projects/federated-reinforcement-learning-cartpole/federatedRLCartpole/rl/rl_main/algorithms_rl')
#from algorithms_rl.PPO_v0 import PPO_v0

# -*- coding: utf-8 -*-
import datetime
import sys
import time

import numpy as np
import random
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/sr42/Projects/federated-reinforcement-learning-cartpole/federatedRLCartpole/rl/rl_main')
import rl_utils

from main_constants import device, PPO_K_EPOCH, GAE_LAMBDA, PPO_EPSILON_CLIP, PPO_VALUE_LOSS_WEIGHT, PPO_ENTROPY_WEIGHT, TRAJECTORY_SAMPLING, TRAJECTORY_LIMIT_SIZE, TRAJECTORY_BATCH_SIZE, LEARNING_RATE


class PPO_v0:
    def __init__(self, env, worker_id, gamma, env_render, logger, verbose):
        self.env = env

        self.worker_id = worker_id

        # discount rate
        self.gamma = gamma

        self.trajectory = []

        # learning rate
        self.learning_rate = LEARNING_RATE

        self.env_render = env_render
        self.logger = logger
        self.verbose = verbose

        self.model = rl_utils.get_rl_model(self.env, self.worker_id)

        self.optimizer = rl_utils.get_optimizer(
            parameters=self.model.parameters(),
            learning_rate=self.learning_rate
        )

    def put_data(self, transition):
        self.trajectory.append(transition)

    def get_trajectory_data(self, sampling=False):
        # print("Before - Trajectory Size: {0}".format(len(self.trajectory)))

        state_lst, action_lst, reward_lst, next_state_lst, prob_action_lst, done_mask_lst = [], [], [], [], [], []
        if sampling:
            sampling_index = random.randrange(0, len(self.trajectory) - TRAJECTORY_BATCH_SIZE + 1)
            trajectory = self.trajectory[sampling_index : sampling_index + TRAJECTORY_BATCH_SIZE]
        else:
            trajectory = self.trajectory

        for transition in trajectory:
            s, a, r, s_prime, prob_a, done = transition

            if type(s) is np.ndarray:
                state_lst.append(s)
            else:
                state_lst.append(s.numpy())

            action_lst.append(a)
            reward_lst.append([r])

            if type(s) is np.ndarray:
                next_state_lst.append(s_prime)
            else:
                next_state_lst.append(s_prime.numpy())

            prob_action_lst.append([prob_a])

            done_mask = 0 if done else 1
            done_mask_lst.append([done_mask])

        state_lst = torch.tensor(state_lst, dtype=torch.float).to(device)
        # action_lst = torch.tensor(action_lst).to(device)
        action_lst = torch.cat(action_lst, 0).to(device)
        reward_lst = torch.tensor(reward_lst).to(device)
        next_state_lst = torch.tensor(next_state_lst, dtype=torch.float).to(device)
        done_mask_lst = torch.tensor(done_mask_lst, dtype=torch.float).to(device)
        prob_action_lst = torch.tensor(prob_action_lst).to(device)

        # print("After - Trajectory Size: {0}".format(len(self.trajectory)))

        # print("state_lst.size()", state_lst.size())
        # print("action_lst.size()", action_lst.size())
        # print("reward_lst.size()", reward_lst.size())
        # print("next_state_lst.size()", next_state_lst.size())
        # print("done_mask_lst.size()", done_mask_lst.size())
        # print("prob_action_lst.size()", prob_action_lst.size())
        return state_lst, action_lst, reward_lst, next_state_lst, done_mask_lst, prob_action_lst

    def train_net(self):

        state_lst, action_lst, reward_lst, next_state_lst, done_mask_lst, prob_action_lst = self.get_trajectory_data()

        loss_sum = 0.0
        for i in range(PPO_K_EPOCH):
            if TRAJECTORY_SAMPLING:
                state_lst, action_lst, reward_lst, next_state_lst, done_mask_lst, prob_action_lst = self.get_trajectory_data(sampling=True)
            else:
                pass
            # print("WORKER: {0} - PPO_K_EPOCH: {1}/{2} - state_lst: {3}".format(self.worker_id, i+1, PPO_K_EPOCH, state_lst.size()))


            state_values = self.model.get_critic_value(state_lst)

            # discount_r_lst = []
            # discounted_reward = 0
            # for r in reversed(reward_lst):
            #     discounted_reward = r + (self.gamma * discounted_reward)
            #     discount_r_lst.insert(0, discounted_reward)
            # discount_r = torch.tensor(discount_r_lst, dtype=torch.float).to(device)
            #
            # # Normalizing the rewards:
            # discount_r = (discount_r - discount_r.mean()) / (discount_r.std() + 1e-5)
            # discount_r = discount_r.unsqueeze(dim=1)
            #
            # advantage = (discount_r - state_values).detach()

            v_target = reward_lst + self.gamma * self.model.get_critic_value(next_state_lst) * done_mask_lst

            delta = v_target - state_values
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for i, delta_t in enumerate(delta[::-1]):
                advantage = self.gamma * GAE_LAMBDA * done_mask_lst[i] * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage_lst = torch.tensor(advantage_lst, device=device, dtype=torch.float)
            advantage_lst = (advantage_lst - advantage.mean() + torch.tensor(1e-6, dtype=torch.float)) / torch.max(
                advantage_lst.std(),
                torch.tensor(1e-6, dtype=torch.float)
            )

            critic_loss = PPO_VALUE_LOSS_WEIGHT * F.smooth_l1_loss(input=state_values, target=v_target.detach())
            # critic_loss = PPO_VALUE_LOSS_WEIGHT * F.smooth_l1_loss(input=state_values, target=discount_r.detach())

            self.optimizer.zero_grad()
            critic_loss.mean().backward()
            self.optimizer.step()

            _, new_prob_action_lst, dist_entropy = self.model.evaluate_for_other_actions(state_lst, action_lst)

            ratio = torch.exp(new_prob_action_lst - prob_action_lst)  # a/b == exp(log(a)-log(b))
            surr1 = ratio * advantage_lst
            surr2 = torch.clamp(ratio, 1 - PPO_EPSILON_CLIP, 1 + PPO_EPSILON_CLIP) * advantage_lst

            # loss = -torch.mean(torch.min(surr1, surr2)) + PPO_VALUE_LOSS_WEIGHT * torch.mean(
            #     torch.mul(advantage_lst, advantage_lst)) - PPO_ENTROPY_WEIGHT * dist_entropy

            actor_loss = - torch.min(surr1, surr2).to(device) - PPO_ENTROPY_WEIGHT * dist_entropy

            self.optimizer.zero_grad()
            actor_loss.mean().backward()
            self.optimizer.step()

            loss = critic_loss + actor_loss

            # print("state_lst_mean: {0}".format(state_lst.mean()))
            # print("next_state_lst_mean: {0}".format(next_state_lst.mean()))
            # print("advantage_lst: {0}".format(advantage_lst[:3]))
            # print("pi: {0}".format(pi[:3]))
            # print("prob: {0}".format(new_prob_action_lst[:3]))
            # print("prob_action_lst: {0}".format(prob_action_lst[:3]))
            # print("new_prob_action_lst: {0}".format(new_prob_action_lst[:3]))
            # print("ratio: {0}".format(ratio[:3]))
            # print("surr1: {0}".format(surr1[:3]))
            # print("surr2: {0}".format(surr2[:3]))
            # print("entropy: {0}".format(entropy[:3]))
            # print("self.model.v(state_lst): {0}".format(self.model.v(state_lst)[:3]))
            # print("v_target: {0}".format(v_target[:3]))
            # print("F.smooth_l1_loss(self.model.v(state_lst), v_target.detach()): {0}".format(F.smooth_l1_loss(self.model.v(state_lst), v_target.detach())))
            # print("loss: {0}".format(loss[:3]))

            # params = self.model.get_parameters()
            # for layer in params:
            #     for name in params[layer]:
            #         print(layer, name, "params[layer][name]", params[layer][name])
            #         break
            #     break
            #
            # print("GRADIENT!!!")


            # actor_fc_named_parameters = self.model.actor_fc_layer.named_parameters()
            # critic_fc_named_parameters = self.model.critic_fc_layer.named_parameters()
            # for name, param in actor_fc_named_parameters:
            #     print("!!!!!!!!!!!!!! - 1 - actor", name)
            #     print(param.grad)
            # for name, param in critic_fc_named_parameters:
            #     print("!!!!!!!!!!!!!! - 2 - critic", name)
            #     print(param.grad)

            # self.optimizer.zero_grad()
            # loss.mean().backward()
            # self.optimizer.step()

            # actor_fc_named_parameters = self.model.actor_fc_layer.named_parameters()
            # critic_fc_named_parameters = self.model.critic_fc_layer.named_parameters()
            # for name, param in actor_fc_named_parameters:
            #     print("!!!!!!!!!!!!!! - 3 - actor", name)
            #     print(param.grad)
            # for name, param in critic_fc_named_parameters:
            #     print("!!!!!!!!!!!!!! - 4 - critic", name)
            #     print(param.grad)

            # self.optimizer.zero_grad()
            # loss.mean().backward()
            # self.optimize_step()

            # grads = self.model.get_gradients_for_current_parameters()
            # for layer in params:
            #     for name in params[layer]:
            #         print(layer, name, "grads[layer][name]", grads[layer][name])
            #         break
            #     break
            #
            #
            #
            # params = self.model.get_parameters()
            # for layer in params:
            #     for name in params[layer]:
            #         print(layer, name, "params[layer][name]", params[layer][name])
            #         break
            #     break

            loss_sum += loss.mean().item()

        self.trajectory.clear()

        gradients = self.model.get_gradients_for_current_parameters()
        return gradients, loss_sum / PPO_K_EPOCH

    def on_episode(self, episode):

        score = 0.0
        number_of_reset_call = 0.0

        if TRAJECTORY_SAMPLING:
            max_trajectory_len = TRAJECTORY_LIMIT_SIZE
        else:
            max_trajectory_len = 0

        while not len(self.trajectory) >= max_trajectory_len:
            done = False
            state = self.env.reset()
            number_of_reset_call += 1.0
            while not done:
                #start_time = datetime.datetime.now()
                if self.env_render:
                    self.env.render()
                action, prob = self.model.act(state)

                next_state, reward, adjusted_reward, done, info = self.env.step(action)
                if "dead" in info.keys():
                    if info["dead"]:
                        self.put_data((state, action, adjusted_reward, next_state, prob, info["dead"]))
                else:
                    self.put_data((state, action, adjusted_reward, next_state, prob, done))

                state = next_state
                score += reward
                #elapsed_time = datetime.datetime.now() - start_time

                #print(elapsed_time, " !!!")

        avrg_score = score / number_of_reset_call
        gradients, loss = self.train_net()
        #print("episode", episode, action)
        return gradients, loss, avrg_score

    def get_parameters(self):
        return self.model.get_parameters()

    def transfer_process(self, parameters, soft_transfer, soft_transfer_tau):
        self.model.transfer_process(parameters, soft_transfer, soft_transfer_tau)


from algorithms_dp.DP_Policy_Iteration import Policy_Iteration
from algorithms_dp.DP_Value_Iteration import Value_Iteration


def get_environment(owner="chief"):
    if ENVIRONMENT_ID == EnvironmentName.QUANSER_SERVO_2:
        client = mqtt.Client(client_id="env_sub_2", transport="TCP")
        env = EnvironmentRIP(mqtt_client=client)

        def __on_connect(client, userdata, flags, rc):
            print("mqtt broker connected with result code " + str(rc), flush=False)
            client.subscribe(topic=MQTT_SUB_FROM_SERVO)
            client.subscribe(topic=MQTT_SUB_MOTOR_LIMIT)
            client.subscribe(topic=MQTT_SUB_RESET_COMPLETE)

        def __on_log(client, userdata, level, buf):
            print(buf)

        def __on_message(client, userdata, msg):
            global PUB_ID

            if msg.topic == MQTT_SUB_FROM_SERVO:
                servo_info = json.loads(msg.payload.decode("utf-8"))
                motor_radian = float(servo_info["motor_radian"])
                motor_velocity = float(servo_info["motor_velocity"])
                pendulum_radian = float(servo_info["pendulum_radian"])
                pendulum_velocity = float(servo_info["pendulum_velocity"])
                pub_id = servo_info["pub_id"]
                env.set_state(motor_radian, motor_velocity, pendulum_radian, pendulum_velocity)

            elif msg.topic == MQTT_SUB_MOTOR_LIMIT:
                info = str(msg.payload.decode("utf-8")).split('|')
                pub_id = info[1]
                if info[0] == "limit_position":
                    env.is_motor_limit = True
                elif info[0] == "reset_complete":
                    env.is_limit_complete = True

            elif msg.topic == MQTT_SUB_RESET_COMPLETE:
                env.is_reset_complete = True
                servo_info = str(msg.payload.decode("utf-8")).split('|')
                motor_radian = float(servo_info[0])
                motor_velocity = float(servo_info[1])
                pendulum_radian = float(servo_info[2])
                pendulum_velocity = float(servo_info[3])
                pub_id = servo_info[4]
                env.set_state(motor_radian, motor_velocity, pendulum_radian, pendulum_velocity)

        if owner == "worker":
            client.on_connect = __on_connect
            client.on_message =  __on_message
            # client.on_log = __on_log

            # client.username_pw_set(username="link", password="0123")
            client.connect(MQTT_SERVER_FOR_RIP, 1883, 3600)

            print("***** Sub thread started!!! *****", flush=False)
            client.loop_start()

    elif ENVIRONMENT_ID == EnvironmentName.CARTPOLE_V0:
        env = CartPole_v0()
    elif ENVIRONMENT_ID == EnvironmentName.CARTPOLE_V1:
        env = CartPole_v1()
    #elif ENVIRONMENT_ID == EnvironmentName.CHASER_V1_MAC or ENVIRONMENT_ID == EnvironmentName.CHASER_V1_WINDOWS:
        #env = Chaser_v1(MY_PLATFORM)
    elif ENVIRONMENT_ID == EnvironmentName.BREAKOUT_DETERMINISTIC_V4:
        env = BreakoutDeterministic_v4()
    elif ENVIRONMENT_ID == EnvironmentName.PENDULUM_V0:
        env = Pendulum_v0()
    elif ENVIRONMENT_ID == EnvironmentName.DRONE_RACING_MAC or ENVIRONMENT_ID == EnvironmentName.DRONE_RACING_WINDOWS:
        env = Drone_Racing(MY_PLATFORM)
    elif ENVIRONMENT_ID == EnvironmentName.GRIDWORLD_V0:
        env = GRIDWORLD_v0()
    elif ENVIRONMENT_ID == EnvironmentName.BLACKJACK_V0:
        env = Blackjack_v0()
    elif ENVIRONMENT_ID == EnvironmentName.FROZENLAKE_V0:
        env = FrozenLake_v0()
    elif ENVIRONMENT_ID == EnvironmentName.INVERTED_DOUBLE_PENDULUM_V2:
        env = InvertedDoublePendulum_v2()
    elif ENVIRONMENT_ID == EnvironmentName.HOPPER_V2:
        env = Hopper_v2()
    elif ENVIRONMENT_ID == EnvironmentName.ANT_V2:
        env = Ant_v2()
    elif ENVIRONMENT_ID == EnvironmentName.HALF_CHEETAH_V2:
        env = HalfCheetah_v2()
    elif ENVIRONMENT_ID == EnvironmentName.SWIMMER_V2:
        env = Swimmer_v2()
    elif ENVIRONMENT_ID == EnvironmentName.REACHER_V2:
        env = Reacher_v2()
    elif ENVIRONMENT_ID == EnvironmentName.HUMANOID_V2:
        env = Humanoid_v2()
    elif ENVIRONMENT_ID == EnvironmentName.HUMANOID_STAND_UP_V2:
        env = HumanoidStandUp_v2()
    elif ENVIRONMENT_ID == EnvironmentName.INVERTED_PENDULUM_V2:
        env = InvertedPendulum_v2()
    elif ENVIRONMENT_ID == EnvironmentName.WALKER_2D_V2:
        env = Walker2D_v2()
    else:
        env = None
    return env


def get_rl_model(env, worker_id):
    if DEEP_LEARNING_MODEL == DeepLearningModelName.ActorCriticMLP or DEEP_LEARNING_MODEL == DeepLearningModelName.ActorCriticCNN:
        model = ActorCriticModel(
            s_size=env.n_states,
            a_size=env.n_actions,
            continuous=env.continuous,
            worker_id=worker_id,
            device=device
        ).to(device)
    elif DEEP_LEARNING_MODEL == DeepLearningModelName.NoModel:
        model = None
    else:
        model = None
    return model


def get_rl_algorithm(env, worker_id=0, logger=False):
    if RL_ALGORITHM == RLAlgorithmName.PPO_V0:
        rl_algorithm = PPO_v0(
            env=env,
            worker_id=worker_id,
            gamma=GAMMA,
            env_render=ENV_RENDER,
            logger=logger,
            verbose=VERBOSE
        )
    # elif RL_ALGORITHM == RLAlgorithmName.DQN_V0:
    #     rl_algorithm = DQN_v0(
    #         env=env,
    #         worker_id=worker_id,
    #         gamma=GAMMA,
    #         env_render=ENV_RENDER,
    #         logger=logger,
    #         verbose=VERBOSE
    #     )
    elif RL_ALGORITHM == RLAlgorithmName.Policy_Iteration:
        rl_algorithm = Policy_Iteration(
            env=env,
            gamma=GAMMA
        )
    elif RL_ALGORITHM == RLAlgorithmName.Value_Iteration:
        rl_algorithm = Value_Iteration(
            env=env,
            gamma=GAMMA
        )
    # elif RL_ALGORITHM == RLAlgorithmName.Monte_Carlo_Control_V0:
    #     rl_algorithm = Monte_Carlo_Control_v0(
    #         env=env,
    #         worker_id=worker_id,
    #         gamma=GAMMA,
    #         env_render=ENV_RENDER,
    #         logger=logger,
    #         verbose=VERBOSE
    #     )
    else:
        rl_algorithm = None

    return rl_algorithm


def get_optimizer(parameters, learning_rate):
    if OPTIMIZER == OptimizerName.ADAM:
        optimizer = optim.Adam(params=parameters, lr=learning_rate)
    elif OPTIMIZER == OptimizerName.NESTEROV:
        optimizer = optim.SGD(params=parameters, lr=learning_rate, nesterov=True, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = None

    return optimizer

# 11785-Project : Deep Reinforcement Learning for Robot Arm Manipulation

This project aims to implement a Deep Reinforcement Learning algorithm in order to move a robot arm to a desired point in cartesian space. OpenAI's Gym toolkit is used to simulate the environment and run experiments in order to train the model and validate the results. The specific environment being used for this project is the [FetchReach-v1](https://gym.openai.com/envs/FetchReach-v1/) environment. 

![The FetchReach-v1 Environment Rendering](/images/fetch_reach_blank.png)

## Organization
The baseline DDPG algorithm can be found in the /DDPG_Baseline directory. This code is based on the DDPG implementation described in [this article](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b) by Chris Yoon, which can also be found in full in [this repository](https://github.com/cyoon1729/Reinforcement-learning).

Modifications to this code including the integration of Hindsight Experience Replay (HER) and Twin-Delayed DDPG (TD3) can all be found in the /Main directory.

## Theory
For an in depth overview of the concepts involved in this project, you can have a look at the Project Report in the /report directory.

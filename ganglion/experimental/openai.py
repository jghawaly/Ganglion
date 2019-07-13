import gym
env = gym.make('BipedalWalker-v2')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        # print(observation/255)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(env.action_space)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()




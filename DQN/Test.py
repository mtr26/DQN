from DQN import Agent, AgentCNN, np
import AtariGames as at
import cv2

def preProcess(image):
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # To grayscale
    #frame = frame[self.crop_dim[0]:self.crop_dim[1], self.crop_dim[2]:self.crop_dim[3]]  # Cut 20 px from top
    frame = cv2.resize(frame, (84, 84))  # Resize
    frame = frame.reshape(84, 84) / 255  # Normalize
    return frame

def empack(arr, ele):
    arr = np.stack((arr[1], arr[2], arr[3], ele))
    return arr

game = at.Game_ATARI(at.PONG)
action_dim = game.inputs
mem_capacity = 1024
batch_capacity = 5
epsilon_decay = 0.9
epsimon_min = 0.01
lr = 0.5
gamma = 0.9

agent = AgentCNN(action_dim, mem_capacity, batch_capacity, epsilon_decay, epsimon_min, lr, gamma)

nbr_episode = int(1e5)


MAX_STEP = 50000
my_step = 1
for epoch in range(nbr_episode):
    obs = game.first_obs
    obs = preProcess(obs)
    obs = np.stack((obs, obs, obs, obs))
    rewards = []
    losses = []
    for step in range(MAX_STEP):
        action = agent.select_action(obs)
        old_obs = obs.copy()
        obs, reward, terminated, truncated, info = game.env.step(action)
        
        if reward < 0:
            reward -= 10
        elif reward > 0:
            reward += 5
        rewards.append(reward)
        obs = preProcess(obs)
        obs = empack(old_obs, obs)
        
        
        agent.push(old_obs, action, reward, obs)
        loss = agent.backward()
        losses.append(loss)

        if terminated or truncated:
            game.env.reset()
            break

        if step % 200 == 0:
            agent.update_weight()
        if my_step % 500 == 0:
            agent.epsilondecrease()

        my_step += 1
    
        if terminated or truncated:
            game.first_obs, info = game.env.reset()
            game.rewards.append(reward)  # Ajoutez cette récompense à la liste des récompenses à la fin de l'épisode
        game.env.render()
    print(f"Episode : {epoch}, cumulative reward : {np.sum(rewards)}, average reward : {np.mean(rewards)}, average loss : {np.mean(losses)}, epsiolon : {agent.epsilon}, steps : {step}")

game.env.close()  # Assurez-vous de fermer l'environnement à la fin
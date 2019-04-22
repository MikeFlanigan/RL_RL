import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import adam
from keras.optimizers import sgd
import matplotlib.pyplot as plt
import time
import json

class environment(object):
    def __init__(self, grid_size=4):
        print('initializing environement...')
        self.grid_size = grid_size
        self.reset()
    
    def update(self, action):
##        print('update the world based on action...')

        # init some stuff...
        game_over = False
        won = False
        reward = 0 # new reward
        
        # map action to useable var
        row_d = 0; col_d = 0;
        if action == 1: row_d = -1
        elif action == 3: row_d = 1
        elif action == 2: col_d = -1
        elif action == 4: col_d = 1

        # check if moves are legal
        if (0 <= self.agent_row + row_d < self.grid_size and 
           0 <= self.agent_col + col_d < self.grid_size):
            
            self.state[self.agent_row,self.agent_col] = 0 # erase old position
            
            self.agent_row += row_d
            self.agent_col += col_d

            # check if the agent moved into the finish square
            if self.state[self.agent_row,self.agent_col] == 3:
                game_over = True
                won = True
                reward += 15 # big reward for winning
                
            self.state[self.agent_row,self.agent_col] = 1 # write new position
        else:
            reward -= 1 # penalize illegal moves 
##            pass # tried to move through a wall

##        reward -= 1 # penalize for existing... pretty ecofriendly reward IMO

        # give reward for moving euc closer... ?
        
##        self.visualize_state()

        return game_over, won, reward
        
    def reset(self):
##        print('resetting state...')
        self.state = np.zeros((self.grid_size,self.grid_size))

##        '''
##        state key: 0 = empty, 1 = agent, 2 = sub reward, 3 = finish, -1 = pit
##        '''
        # enforce a static agent starting position for now
        self.agent_row = self.grid_size-1
        self.agent_col = 0
        self.state[self.agent_row,self.agent_col] = 1

        # enforce a static end position for now
        self.state[0,self.grid_size-1] = 3

##        self.visualize_state()
        
    def visualize_state(self):
        # for now just printing the state in a viewable array
##        pass
        print(self.state)



# ya cool to put this into a __if main style so this program can have a tester script too...
# parameters
world_size = 5

# create model
hidden_size = 2*world_size**2
num_actions = 4
model = Sequential()
model.add(Dense(hidden_size, input_shape=(world_size**2,), activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(num_actions, activation='softmax')) # need to upgrade tensorflow
model.compile(loss="mse", optimizer=adam()) # since no params given to adam means its using all paper defaults
##model.compile(sgd(lr=.01), "mse")

model_name_to_load = "model.h5"
model_name_to_save = "model.h5"

# If you want to continue training from a previous model, just uncomment the line bellow
model.load_weights(model_name_to_load)
    
# create the block world environment
static_bw = environment(world_size)

human_control = False
episodes = 15000
epoch_score_hist = []
epoch_loss_hist = []
wins = 0
losses = 0
print('running...')

try:
    for e in range(episodes):
        if np.mod(e,100) == 0: print('epoch :',e,' wins: ',wins,' losses: ',losses)
        game_over = False
        total_score = 0
        experience = np.reshape(static_bw.state.flatten(),(1,world_size**2))
        act_mem = []
        loss = 0
        greedy_epsilon = 0.01
        move_count = 0
        move_limit = 20
        static_bw.reset()
        while not game_over:

            # get action
            action = 0
            if human_control:
                static_bw.visualize_state()
                Q_pred = model.predict(np.reshape(static_bw.state.flatten(),(1,world_size**2)))
                print('model prediction:',Q_pred)
                print('model suggestion:',np.argmax(Q_pred)+1)
                key = input('user command: ')
                if key == 'w': action = 1 # go forward
                elif key == 'a': action = 2 # go left
                elif key == 's': action = 3 # go backwards
                elif key == 'd': action = 4 # go right
                elif key == 'q': game_over = True
            else:
                Q_pred = model.predict(np.reshape(static_bw.state.flatten(),(1,world_size**2)))
                if greedy_epsilon < np.random.rand(1)[0]:
                    action = np.argmax(Q_pred)
                else:
                    action = np.random.randint(num_actions)
                action += 1 # integer offset

            act_mem.append(action)
            
            # update state based on action
            terminate, won, reward = static_bw.update(action)
            total_score += reward

            move_count += 1 
            if move_count >= move_limit:
                terminate = True
                losses += 1
    ##            print('Game over, move count reached')
            if terminate:
                game_over = True
                if won: wins += 1
            else:
                experience = np.concatenate((experience,np.reshape(static_bw.state.flatten(),(1,world_size**2))),0)

##            if e > 10:
##                print(Q_pred)
##                static_bw.visualize_state()
##                time.sleep(0.1)
            
        # create targets
        tar_Qs = np.zeros((len(act_mem),4))
        for a in range(len(act_mem)):
            tar_Qs[a,act_mem[a]-1] = total_score # not sure if this should be normalized by something...

        loss += model.train_on_batch(experience, tar_Qs)
        
    ##    print('Game over! hopefully you won...')
    ##    print('Episode: ',e,' loss: ',loss,' Score: ',total_score)

        epoch_score_hist.append(total_score)
        epoch_loss_hist.append(loss)
        
except KeyboardInterrupt:
    pass

# Save trained model weights and architecture, this will be used by the visualization code
##model.save_weights(model_name_to_save, overwrite=True)
##with open("model.json", "w") as outfile:
##    json.dump(model.to_json(), outfile)
        
plt.subplot(2, 1, 1)
plt.plot(epoch_score_hist)
plt.ylabel('score hist')

plt.subplot(2, 1, 2)
plt.plot(epoch_loss_hist)
plt.ylabel('loss hist')
plt.show()

        
    

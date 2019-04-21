import json
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
from RL_catcher import Catch


if __name__ == "__main__":
    # Make sure this grid size matches the value used for training
    grid_size = 10

    with open("model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
##    model.load_weights("model_min_training.h5")
    model.load_weights("model2.h5")
    model.compile("sgd", "mse")

    # Define environment, game
    env = Catch(grid_size)
    c = 0

    scores = []
    for MC in range(50):
        score = 0
        for e in range(10):
            loss = 0.
            env.reset()
            game_over = False
            # get initial input
            input_t = env.observe()

            plt.ion()
            plt.imshow(input_t.reshape((grid_size,)*2),
                       interpolation='none', cmap='gray')
    ##        plt.savefig("%03d.png" % c)
            c += 1
        
            while not game_over:
                input_tm1 = input_t

                # get next action
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

                # apply action, get rewards and new state
                input_t, reward, game_over = env.act(action)
                if reward == 1: score += 1
                plt.imshow(input_t.reshape((grid_size,)*2),
                           interpolation='none', cmap='gray')
    ##                                              plt.savefig("%03d.png" % c)
    ##                                              plt.draw()
                plt.show()

                plt.pause(0.0001)
                plt.clf()
                c += 1
        scores.append(score)
        print('Monte Carlo run score: ',score)

plt.hist(scores,[x-0.5 for x in range(1,12)],edgecolor="k")
plt.xlabel('fruits of 10 caught')
plt.ylabel('number of MC runs that got that #')
plt.show()

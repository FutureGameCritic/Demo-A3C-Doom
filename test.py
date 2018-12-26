from random import choice
from time import sleep

from game import doom
game, actions = doom()

game.new_episode()

while not game.is_episode_finished():
    # Gets the state
    state = game.get_state()

    # Which consists of:
    n = state.number
    vars = state.game_variables
    screen_buf = state.screen_buffer
    depth_buf = state.depth_buffer
    labels_buf = state.labels_buffer
    automap_buf = state.automap_buffer
    labels = state.labels

    # Games variables can be also accessed via:
    #game.get_game_variable(GameVariable.AMMO2)

    # Makes a random action and get remember reward.
    r = game.make_action(choice(actions))

    # Makes a "prolonged" action and skip frames:
    # skiprate = 4
    # r = game.make_action(choice(actions), skiprate)

    # The same could be achieved with:
    # game.set_action(choice(actions))
    # game.advance_action(skiprate)
    # r = game.get_last_reward()

    # Prints state's game variables and reward.
    print("State #" + str(n))
    print("Game variables:", vars)
    print("Reward:", r)
    print("=====================")
    sleep_time = 1.0
    if sleep_time > 0:
        sleep(sleep_time)

game.close()
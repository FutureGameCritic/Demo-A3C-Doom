def doom(hparams):
    import vizdoom as vzd
    game = vzd.DoomGame()
    # you can change another doom-world
    game.set_doom_scenario_path('./scenarios/basic.wad')
    game.set_doom_map("map01")

    # Sets resolution. Default is 320X240
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)

    # Sets the screen buffer format. Not used here but now you can change it. Defalut is CRCGCB.
    game.set_screen_format(vzd.ScreenFormat.GRAY8)

    # Enables depth buffer.
    game.set_depth_buffer_enabled(False)

    # Enables labeling of in game objects labeling.
    game.set_labels_buffer_enabled(True)

    # Enables buffer with top down map of the current episode/level.
    game.set_automap_buffer_enabled(False)

    # Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
    if hparams.fast_train is False:
        game.set_render_hud(False)
        game.set_render_minimal_hud(False)  # If hud is enabled
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)  # Bullet holes and blood on the walls
        game.set_render_particles(False)
        game.set_render_effects_sprites(False)  # Smoke and blood
        game.set_render_messages(False)  # In-game messages
        game.set_render_corpses(False)
        game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items
    else:
        game.set_window_visible(False)


    # Adds buttons that will be allowed.
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.add_available_button(vzd.Button.ATTACK)

    # Causes episodes to finish after 200 tics (actions)
    game.set_episode_timeout(hparams.n_step)

    # Makes episodes start after 10 tics (~after raising the weapon)
    game.set_episode_start_time(10)

    # Turns on the sound. (turned off by default)
    game.set_sound_enabled(False)

    # Sets the livin reward (for each move) to -1
    game.set_living_reward(-1)

    # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
    game.set_mode(vzd.Mode.PLAYER)

    # Initialize the game. Further configuration won't take any effect from now on.
    game.init()

    # Define some actions. Each list entry corresponds to declared buttons:
    # MOVE_LEFT, MOVE_RIGHT, ATTACK
    # game.get_available_buttons_size() can be used to check the number of available buttons.
    # 5 more combinations are naturally possible but only 3 are included for transparency when watching.
    actions = [[True, False, False], [False, True, False], [False, False, True]]

    return game, actions
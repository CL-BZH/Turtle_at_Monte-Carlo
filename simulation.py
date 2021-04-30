#!/usr/bin/env python3

from mc_turtle import *

#Landmark 0: (121.68234460415977,34.197601273212484)
#Landmark 1: (-52.99160487358641,-124.4965354750137)
#Landmark 2: (-24.371889641058104,-0.36786214345718804)
#Landmark 3: (-125.91402231540933,19.61204475881749)


def main(args=None):
    # Set the scene
    screen_width = 450
    screen_height = 450
    x_lim = 160
    y_lim = 160
    nb_landmarks = 4
    
    landmarks = [Landmark(-39.9, 1.64, 0, '#ffd700')]
    landmarks += [Landmark(59.3, -39.8, 1, '#ffa500')]
    landmarks += [Landmark(62.3, 117.1, 2, '#8a2be2')]
    landmarks += [Landmark(-58.2, 103.15, 3, '#ff7f50')]

    min_dist_between_landmarks = 100
    
    floorplan = Floorplan(screen_width, screen_height,
                          x_lim, y_lim, nb_landmarks)
    floorplan.set_screen("Tracking the turtle")
    floorplan.draw_bounderies()
    floorplan.draw_landmarks(min_dist_between_landmarks, landmarks)
    #floorplan.draw_landmarks(min_dist_between_landmarks)
    
    # Intentiate a simulation
    sim = Simulation(floorplan, 256)
    # Keyboard's key binding:
    # Pause when "space" key is pressed
    # Quit when "q" key is pressed
    onkeys = {"space": sim.pause, "q": sim.end}
    floorplan.bind_keys(onkeys)
    # Init the particles
    sim.init_particles()
    # And run!
    #sim.run(with_error = False)
    sim.run(with_error = True)

if __name__ == "__main__":
    main()

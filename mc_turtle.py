#!/usr/bin/env python3

import turtle
from turtle import Turtle
import time
import random as rd
import math
import numpy as np
import bisect
import itertools


screen_width = 450
screen_height = 450
x_lim = 160
y_lim = 160
forward_step = 4
move_angle = np.pi/60

nb_particles = 196
assert nb_particles > 0
nb_landmarks = 4
assert nb_landmarks > 0

min_dist_between_landmarks = 100 # Should be adapted to the number of landmark
                                 # and the size of the map

class Landmark:
    def __init__(self, x, y, Id='None', color='black'):
        self.x = x
        self.y = y
        self.color = color
        self.id = Id


class Measurement:
    def __init__(self, distance, angle, Id='None'):
        self.distance = distance
        self.angle = angle
        self.landmark_id = Id
        
    def __str__(self):
        return (f"Landmark {self.landmark_id}, distance: {self.distance}, "
                f"angle: {self.angle}")

        
class Robot(Turtle):
    def __init__(self, shape, color):
        super().__init__(shape=shape)
        #self.up()
        self.color(color)
        self.radians()
        self.speed(10)
        self.max_meas_range = 200
        self.measurements = []
        self.theta_sigma = 0.2
        self.step_sigma = 0.5
        
    def set(self, x, y, theta):
         self.setposition(x, y)
         self.setheading(theta)
         
    def move(self, step, theta):
        self.right(theta)  
        self.forward(step)
        #self.x += forward_step * np.cos(theta)
        #self.y += forward_step * np.sin(theta)
        #self.theta += theta
        #self.set(self.x, self.y, self.theta)

    def move_with_error(self, step, theta):
        theta = rd.gauss(theta, self.theta_sigma)
        step = rd.gauss(step, self.step_sigma)
        self.move(step, theta)
        
    # Measurement is perfectly accurate even though
    # we are assuming it isn't.
    def measure(self, landmarks, trace=False):
        self.measurements.clear()
        x, y = self.pos()
        theta = self.heading()
        for Id, landmark in enumerate(landmarks):
            dx = x - landmark.x
            dy = y - landmark.y
            distance = math.sqrt(dx*dx + dy*dy)
            if distance < self.max_meas_range:
                angle = math.atan2(dy, dx) - theta
                while angle >= 2*math.pi:
                    angle -= 2*math.pi
                while angle <= -2*math.pi:
                    angle += 2*math.pi
                self.measurements += [Measurement(distance, angle, Id)]
                if trace:
                   print(self.measurements[-1])



class Particle(Robot):
    def __init__(self):
        super().__init__("arrow", "blue")
        self.shapesize(0.5, 0.5, 0.5)
        self.penup()
        # Standard deviation for the predict (redefine step_sigma
        # and theta_sigma of the Robot)
        self.step_sigma = 1.2
        self.theta_sigma = 0.2
        
        self.weight = 0.0
        # Standard deviation for weighting
        self.distance_sigma = 16
        self.angle_sigma = 0.1

        
    def predict(self, step, theta):
        step = rd.gauss(step, self.step_sigma)
        theta = rd.gauss(theta, self.theta_sigma)
        self.move(step, theta)

    def probability_density_function(self, mu, sigma, x):
        return np.exp(-((mu-x)**2)/(2*sigma**2)) # No need to normalize

    def update_weight(self, robot_measurements):
        self.weight = 0.0
        x, y = self.pos()
        if ((x > x_lim + 20) or (x < -x_lim - 20) or
            (x > y_lim + 20) or (x < -y_lim - 20)):
            # Particle is outside bounds
            return
        
        for robot_meas in robot_measurements:
            robot_dist = robot_meas.distance
            robot_angle = robot_meas.angle
            best_weight = 0.0
            distance_weight = 0.0
            angle_weight = 0.0
            #print(f"Robot meas for landmark {robot_meas.landmark_id}")
            selected_landmark = 0
            for meas in self.measurements:
                # Just select the particle measurement that best match
                # the current robot measurement.
                # Note: that means that some measurement can be selected
                # more than once
                particle_dist = meas.distance
                particle_angle = meas.angle
                diff_angle = abs(robot_angle - particle_angle)
                if diff_angle > np.pi:
                    diff_angle = diff_angle - 2*np.pi
                distance_weight = self.probability_density_function(robot_dist,
                                                           self.distance_sigma,
                                                           particle_dist)
                #print(f"distance weight: {distance_weight}")
                angle_weight = self.probability_density_function(0,
                                                                 self.angle_sigma,
                                                                 diff_angle)
                #print(f"angle weight: {angle_weight}")
                weight = distance_weight * angle_weight
                if weight > best_weight:
                    best_weight = weight
                    selected_landmark = meas.landmark_id
            #print(f"\tParticle meas landmark {selected_landmark}")        
            self.weight += best_weight

            
    def update_weight_0(self, robot_measurements):
        self.weight = 0
        best_weight = 0
        length = min(len(robot_measurements), len(self.measurements))
        robot_measurements.sort(key=lambda x: (x.distance, x.angle))
        print(f"Robot meas for landmarks: ", end='')
        for meas in robot_measurements:
            print(f"{meas.landmark_id} ", end='')
        print('\n')    
        self.measurements.sort(key=lambda x: (x.distance, x.angle))
        print(f"Measurements length: {length}")
        robot_measurements = robot_measurements[:length]
        self.measurements = self.measurements[:length]
        best_match = self.measurements
        for p_measurements in list(itertools.permutations(self.measurements)):
            for i in range(len(p_measurements)):
                delta_distance = (robot_measurements[i].distance
                                  - p_measurements[i].distance)
                weight = self.probability_density_function(0,
                                                           self.distance_sigma,
                                                           delta_distance)
                diff_angle = abs(robot_measurements[i].angle
                                 - p_measurements[i].angle)
                if diff_angle > np.pi:
                    diff_angle = diff_angle - 2*np.pi
                weight *= self.probability_density_function(0,
                                                            self.angle_sigma,
                                                            diff_angle)
                if weight > best_weight:
                    best_weight = weight
                    best_match = p_measurements
        #best_weight /= length
        self.weight = best_weight
        print(f"Particle best match landmarks: ", end='')
        for meas in best_match:
            print(f"{meas.landmark_id} ", end='')
        print('\n')
        
class Env:
    def __init__(self):
        self.wn = turtle.Screen()
        self.wn.setup(screen_width, screen_height)
        self.wn.title("Tracking Turtle")
        # Pause when spacebar is pressed
        self.wn.onkey(self.pause, "space")
        # Quit when 'q' key is pressed
        self.wn.onkey(self.end, "q")

        # Create the robot
        self.robot = Robot("turtle", "lime")
        
        # The particles
        self.particles = []
        self.resampling_count = 0

        # Draw the bounderies for the particles
        self.robot.hideturtle()
        self.wn.tracer(0)
        self.robot.up()
        self.robot.goto(-x_lim - 2, -y_lim - 2)
        self.robot.color('red')
        self.robot.setheading(np.pi/2)
        self.robot.down()
        for _ in range(2):
            self.robot.forward(2*x_lim + 4)
            self.robot.right(np.pi/2)
            self.robot.forward(2*y_lim + 4)
            self.robot.right(np.pi/2)
        self.robot.up()
        self.robot.color('lime')    
        self.robot.goto(0, 0)
        self.robot.showturtle()
        self.robot.down()
        self.wn.update()
        self.wn.tracer(1)

        # Draw the landmarks
        self.robot.hideturtle()
        self.wn.tracer(0)
        self.robot.color('black') 
        lim_x = 0.75 * x_lim
        lim_y = 0.75 * y_lim
        self.landmarks = []
        x = rd.uniform(-lim_x, lim_x)
        y = rd.uniform(-lim_y, lim_y)
        self.landmarks += [Landmark(x,y, 0)]
        print(f"Landmark 0: ({x},{y})")
        self.robot.up()
        self.robot.goto(x, y)
        self.robot.down()
        self.robot.dot(10)
        for i in range(1, nb_landmarks):
            self.robot.up()
            while True:
                x = rd.uniform(-lim_x, lim_x)
                y = rd.uniform(-lim_y, lim_y)
                min_dist = 1000
                for landmark in self.landmarks:
                    dist = np.sqrt((x-landmark.x)**2 + (y-landmark.y)**2)
                    if dist < min_dist:
                        min_dist = dist
                if min_dist > min_dist_between_landmarks:
                    break
            self.landmarks += [Landmark(x,y, i)]    
            self.robot.goto(x, y)
            self.robot.down()
            self.robot.dot(10)
            print(f"Landmark {i}: ({x},{y})")
        self.robot.up()
        self.robot.color('lime')    
        self.robot.goto(0, 0)
        self.robot.showturtle()
        self.robot.down()
        self.wn.update()
        self.wn.tracer(1)    
         
    def spread_particles(self):
        for particle in self.particles:
            particle.reset()
        self.particles.clear()
        x = 2 * x_lim
        y = 2 * y_lim
        si = (x * y) / nb_particles
        xi = np.sqrt(si * x / y)
        yi = xi * y / x
        x0 = rd.uniform(-x_lim, -x_lim + xi)
        y0 = rd.uniform(-y_lim, -y_lim + yi)
        particles_count = 0
        for i in range(int(round((x/xi)))): 
            for j in range(int(round((y/yi)))):
                particle = Particle()
                pos_x = x0 + i * xi
                pos_y = y0 + j * yi
                theta = rd.uniform(0, math.pi * 2)
                particle.set(pos_x, pos_y, theta)
                particle.showturtle()
                self.particles += [particle]
                particles_count += 1
        print(f"Total number of particle: {particles_count}")
            
        
    def init_particles(self):
        self.wn.tracer(0)
        self.spread_particles()
        self.wn.tracer(1)
        
    def move(self):
        self.wn.tracer(0)
        x, y = self.robot.pos()
        if x**2 + y**2 < 0.1:
            global move_angle
            move_angle *= -1
        self.robot.move(forward_step, move_angle)   
        for particle in self.particles:
            particle.predict(forward_step, move_angle)
        self.wn.tracer(1)

    def measure(self):
        self.wn.tracer(0)
        self.robot.measure(self.landmarks, False)
        self.wn.tracer(1)

    def updates_particles_weights(self):
        self.wn.tracer(0)
        for particle in self.particles:
            particle.measure(self.landmarks, False)
            particle.update_weight(self.robot.measurements)
        self.wn.tracer(1)
        
    def resample_particles(self):
        """
        Perform systematic resampling and if the max weight is too low
        then redistribute the particles uniformly
        """
        self.wn.tracer(0)
        
        self.resampling_count += 1
        max_weight = 0
    
        resampled_particles = []

        cumulative_weights = []
        total_weight = 0
        for particle in self.particles:
            weight = particle.weight
            if weight > max_weight:
                max_weight = weight
            total_weight += weight
            cumulative_weights.append(total_weight)

        print(f"Resampling count: {self.resampling_count}, Max weight: {max_weight}")
        if ((self.resampling_count > 6 and max_weight < 0.95) or
            (self.resampling_count > 12 and max_weight < 2)):
            print(f"+Spread particles+")
            self.spread_particles()
            self.resampling_count = 0
            return

        if max_weight > 2.6:
            # Reset
            self.resampling_count = 0
            #print(f"Converged. max weigth: {max_weight}")
            
        interval_length = total_weight / nb_particles
        s = rd.uniform(0, interval_length)

        for k in range(nb_particles):
            u = s + k * interval_length
            lower_bound_idx = bisect.bisect_left(cumulative_weights, u)
            particle = self.particles[lower_bound_idx]
            x, y = particle.pos()
            x += rd.gauss(0, particle.step_sigma)
            y += rd.gauss(0, particle.step_sigma)
            theta = particle.heading() + rd.gauss(0, particle.theta_sigma)
            new_particle = Particle()
            new_particle.set(x, y, theta)
            resampled_particles.append(new_particle)

        for particle in self.particles:
            particle.reset()
        self.particles.clear()
        self.particles = resampled_particles[:] 
        self.wn.tracer(1)
        
    def run(self):
        self.status = 'running'
        self.wn.listen()
        for particle in self.particles:
            particle.showturtle()
        # Robot go round and round
        while True:
            self.move()
            self.measure()
            self.updates_particles_weights()
            self.resample_particles()
            self.pausing()
    
    def pausing(self):
        while self.status == 'pause':
            time.sleep(0.5)
            self.wn.update()
            
    def pause(self):
        if (self.status == 'running'):
            self.status = 'pause'
        else: # (self.tatus == 'pause')
            self.status = 'running'
        print(self.status)

    def end(self):
        self.wn.bye()

        
def main(args=None):
    sim = Env()
    sim.init_particles()
    sim.run()

if __name__ == "__main__":
    main()

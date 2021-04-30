#!/usr/bin/env python3

import turtle
from turtle import Turtle
import time
import random as rd
import math
import numpy as np
import bisect
import itertools


class Landmark:
    def __init__(self, x, y, Id='None', color='black'):
        self.x = x
        self.y = y
        self.color = color
        self.id = Id

    def draw(self, wn):
        t = turtle.Turtle()
        t.radians()
        t.hideturtle()
        t.up()
        t.goto(self.x, self.y)
        t.down()
        t.dot(10)
        wn._turtles.remove(t)
        
class Floorplan:
    def __init__(self, screen_width=450, screen_height=450,
                 x_lim=160, y_lim=160, nb_landmarks=4):      
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.x_lim = x_lim
        self.y_lim = y_lim
        assert nb_landmarks > 0
        self.nb_landmarks = nb_landmarks
        self.landmarks = []
        
    def set_screen(self, title): 
        self.wn = turtle.Screen()
        self.wn.setup(self.screen_width, self.screen_height)
        self.wn.title(title)

    def bind_keys(self, key_bindings):
        for key, function in key_bindings.items():
            self.wn.onkey(function, key)

    def is_within_bounderies(self, x, y, margin=0):
        x_lim = self.x_lim + margin
        y_lim = self.y_lim + margin
        if ((x < -x_lim) or (x > x_lim) or (y < -y_lim) or (y > y_lim)):
            return False
        return True
    
    def remove_turtle(self, t):
        """
        Delete the turtle t.
        """
        self.wn._turtles.remove(t)
        
    def draw_bounderies(self):
        # Draw the bounderies
        self.wn.tracer(0)
        t = turtle.Turtle()
        t.radians()
        t.hideturtle()
        t.up()
        t.goto(-self.x_lim - 2, -self.y_lim - 2)
        print(f"boundery start: {self.x_lim}, {self.y_lim}")
        t.color('red')
        t.setheading(np.pi/2)
        t.down()
        for _ in range(2):
            t.forward(2*self.x_lim + 4)
            print(f"boundery forward: {2*self.x_lim + 4}")
            t.right(np.pi/2)
            t.forward(2*self.y_lim + 4)
            t.right(np.pi/2)
        self.wn.update()
        self.remove_turtle(t)
        self.wn.tracer(1)

    def draw_landmarks(self, min_spacing=0, landmarks=None):
        # Draw the landmarks
        self.wn.tracer(0)
        if landmarks == None:
            lim_x = 0.85 * self.x_lim
            lim_y = 0.85 * self.y_lim
            x = rd.uniform(-lim_x, lim_x)
            y = rd.uniform(-lim_y, lim_y)
            self.landmarks += [Landmark(x,y, 0)]
            print(f"Landmark 0: ({x},{y})")
            self.landmarks[0].draw(self.wn)
            for i in range(1, self.nb_landmarks):
                while True:
                    x = rd.uniform(-lim_x, lim_x)
                    y = rd.uniform(-lim_y, lim_y)
                    min_dist = 1000
                    for landmark in self.landmarks:
                        dist = np.sqrt((x-landmark.x)**2 + (y-landmark.y)**2)
                        if dist < min_dist:
                            min_dist = dist
                    # Enforce a minimum space between landmarks
                    if min_dist > min_spacing:
                        break
                self.landmarks += [Landmark(x,y, i)]    
                self.landmarks[-1].draw(self.wn)
                print(f"Landmark {i}: ({x},{y})")
        else:
            for landmark in landmarks:
                landmark.draw(self.wn)
                self.landmarks += [landmark]  
        self.wn.tracer(1)

        
class Measurement:
    def __init__(self, distance, angle, Id='None'):
        self.distance = distance
        self.angle = angle
        self.landmark_id = Id
        
    def __str__(self):
        return (f"Landmark {self.landmark_id}, distance: {self.distance}, "
                f"angle: {self.angle}")

        
class Robot(Turtle):
    def __init__(self, shape, color=None):
        super().__init__(shape=shape)
        if color is not None:
            self.color(color)
        self.radians()
        self.speed(10)
        self.max_meas_range = 200
        self.measurements = []
        self.theta_sigma = 0.2
        self.step_sigma = 0.5
        self.x, self.y = self.pos()
        self.theta = self.heading()
        
    def set(self, x, y, theta):
         self.setposition(x, y)
         self.setheading(theta)
         
    def move(self, step, theta, inside_bounderies):
        #self.right(theta)  
        #self.forward(step)
        self.x, self.y = self.pos()
        theta = self.heading() + theta
        x = self.x + step * np.cos(theta)
        y = self.y + step * np.sin(theta)
        if (inside_bounderies(x, y) == False):
            theta += np.pi/2
            x = self.x + step * np.cos(theta)
            y = self.y + step * np.sin(theta)
        self.x, self.y, self.theta = x, y, theta
        self.goto(self.x, self.y)
        self.setheading(self.theta)


    def move_with_error(self, step, theta, inside_bounderies):
        theta = rd.gauss(theta, self.theta_sigma)
        step = rd.gauss(step, self.step_sigma)
        self.move(step, theta, inside_bounderies)
        
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
    def __init__(self, floorplan):
        super().__init__("arrow", "blue")
        self.floorplan = floorplan
        self.shapesize(0.5, 0.5, 0.5)
        self.fillcolor("")
        self.penup()
        # Standard deviation for the predict (redefine step_sigma
        # and theta_sigma of the Robot)
        self.step_sigma = 1.1
        self.theta_sigma = 0.2
        
        self.weight = 0.0
        # Standard deviation for weighting
        self.distance_sigma = 20 #16
        self.angle_sigma = 0.1

    def move(self, step, theta):
        """
        Redefine the robot move() function since for particles
        there is no change of move for the bounderies.
        When a particle cross too much a boundery its weight will
        be set to 0 in update_weight().
        """
        self.x, self.y = self.pos()
        self.theta = self.heading() + theta
        self.x += step * np.cos(self.theta)
        self.y += step * np.sin(self.theta)
        self.goto(self.x, self.y)
        self.setheading(self.theta)
            
    def predict(self, step, theta):
        step = rd.gauss(step, self.step_sigma)
        theta = rd.gauss(theta, self.theta_sigma)
        self.move(step, theta)

    def probability_density_function(self, mu, sigma, x):
        return np.exp(-((mu-x)**2)/(2*sigma**2)) # No need to normalize

    def update_weight(self, robot_measurements):
        self.weight = 0.0
        x, y = self.pos()
        if self.floorplan.is_within_bounderies(x, y, 10) == False:
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

 
        
class Simulation:
    def __init__(self, floorplan, nb_particles):
        self.floorplan = floorplan
        
        # Number of particles
        assert nb_particles > 0
        self.nb_particles = nb_particles
        
        # Create the robot
        self.robot = Robot("turtle", "lime")
        self.robot.color('lime')    
        self.robot.goto(0, 0)
        self.robot.showturtle()
        self.robot.down()
        self.floorplan.wn.update()
        
        # The particles
        self.particles = []
        self.resampling_count = 0
         
    def spread_particles(self):
        for particle in self.particles:
            particle.reset()
            self.floorplan.remove_turtle(particle)
        self.particles.clear()
        x = 2 * self.floorplan.x_lim
        y = 2 * self.floorplan.y_lim
        si = (x * y) / self.nb_particles
        xi = np.sqrt(si * x / y)
        yi = xi * y / x
        x0 = rd.uniform(-self.floorplan.x_lim, -self.floorplan.x_lim + xi)
        y0 = rd.uniform(-self.floorplan.y_lim, -self.floorplan.y_lim + yi)
        particles_count = 0
        for i in range(int(round((x/xi)))): 
            for j in range(int(round((y/yi)))):
                particle = Particle(self.floorplan)
                pos_x = x0 + i * xi
                pos_y = y0 + j * yi
                theta = rd.uniform(0, math.pi * 2)
                particle.set(pos_x, pos_y, theta)
                particle.showturtle()
                self.particles += [particle]
                particles_count += 1
        # Adjust the number of particles
        self.nb_particles = particles_count
        print(f"Total number of particle: {self.nb_particles}")
       
        
    def init_particles(self):
        self.floorplan.wn.tracer(0)
        self.spread_particles()
        self.floorplan.wn.tracer(1)
        
    def move(self, forward_step, move_angle, with_error=True):
        """
        Note: replace move_with_error() below with move() to move
        without error.
        """
        self.floorplan.wn.tracer(0)
        x, y = self.robot.pos()
        heading = self.robot.heading()
        if with_error == True:
            self.robot.move_with_error(forward_step, move_angle,
                                       self.floorplan.is_within_bounderies)
        else:
            self.robot.move(forward_step, move_angle,
                            self.floorplan.is_within_bounderies)
        # Update the particles (predict)
        x_new, y_new = self.robot.pos()
        dx = x_new-x
        dy = y_new-y
        angle = np.pi/2
        if abs(x_new-x) > 0.0001:
            angle = math.atan2(dy, dx) - heading
            while angle >= 2*np.pi:
                angle -= 2*np.pi
            while angle <= -2*np.pi:
                angle += 2*np.pi    
        step = np.sqrt(dx*dx + dy*dy)
        for particle in self.particles:
            particle.predict(step, angle)
        self.floorplan.wn.tracer(1)

    def measure(self):
        self.floorplan.wn.tracer(0)
        self.robot.measure(self.floorplan.landmarks, False)
        self.floorplan.wn.tracer(1)

    def updates_particles_weights(self):
        self.floorplan.wn.tracer(0)
        for particle in self.particles:
            particle.measure(self.floorplan.landmarks, False)
            particle.update_weight(self.robot.measurements)
        self.floorplan.wn.tracer(1)
        
    def resample_particles(self):
        """
        Perform systematic resampling and if the max weight is too low
        then redistribute the particles uniformly
        """
        self.floorplan.wn.tracer(0)
        
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
        if max_weight < 0.95:
            self.resampling_count += 1
            if self.resampling_count > 7:
                print(f"+Spread particles+")
                self.spread_particles()
                self.resampling_count = 0
                return
        elif max_weight < 1.9:
            self.resampling_count += 1
            if self.resampling_count > 14:
                print(f"+Spread particles+")
                self.spread_particles()
                self.resampling_count = 0
                return
        elif max_weight < 2.8:
            self.resampling_count += 1
            if self.resampling_count > 28:
                print(f"+Spread particles+")
                self.spread_particles()
                self.resampling_count = 0
                return
        else:
            self.resampling_count = 0
            #print(f"Converged. max weigth: {max_weight}")
            
        interval_length = total_weight / self.nb_particles
        s = rd.uniform(0, interval_length)

        for k in range(self.nb_particles):
            u = s + k * interval_length
            lower_bound_idx = bisect.bisect_left(cumulative_weights, u)
            particle = self.particles[lower_bound_idx]
            x, y = particle.pos()
            x += rd.gauss(0, particle.step_sigma)
            y += rd.gauss(0, particle.step_sigma)
            theta = particle.heading() + rd.gauss(0, particle.theta_sigma)
            new_particle = Particle(self.floorplan)
            new_particle.set(x, y, theta)
            resampled_particles.append(new_particle)

        for particle in self.particles:
            particle.reset()
            self.floorplan.remove_turtle(particle)
        self.particles.clear()
        self.particles = resampled_particles 
        self.floorplan.wn.tracer(1)

        
    def run(self, with_error):
        self.with_error = with_error
        self.status = 'running'
        self.floorplan.wn.listen()
        for particle in self.particles:
            particle.showturtle()    
        self.robot.setheading(0)
        # The trajectory that the robot would do if move
        # without error is an 8.
        # Step and angle for looping on a 8 shape:
        step = 4
        angle = np.pi/60
        while True:
            self.move(step, angle, self.with_error)
            self.measure()
            self.updates_particles_weights()
            self.resample_particles()
            self.pausing()
            # Second loop of the 8 shape (when move without error)
            if self.with_error == False:
                x, y = self.robot.pos()
                if x**2 + y**2 < 0.1:
                    angle *= -1
    
    def pausing(self):
        while self.status == 'pause':
            time.sleep(0.5)
            self.floorplan.wn.update()
            
    def pause(self):
        if (self.status == 'running'):
            self.status = 'pause'
        else: # (self.tatus == 'pause')
            self.status = 'running'
        print(self.status)

    def end(self):
        self.floorplan.wn.bye()


#!/usr/bin/env python3

import turtle
from turtle import Turtle
import time
import random as rd
import numpy as np
import bisect
import itertools

from math import cos, sin, atan2, sqrt, pi

class Landmark:
    def __init__(self, x, y, Id='None', color='black'):
        self.x = x
        self.y = y
        self.color = color
        self.id = Id

    def draw(self, wn):
        t = turtle.Turtle()
        t.color(self.color)
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
        # Draw the (fake) bounderies
        self.wn.tracer(0)
        t = turtle.Turtle()
        t.radians()
        t.hideturtle()
        t.up()
        t.goto(-self.x_lim - 2, -self.y_lim - 2)
        t.color('lightgrey')
        t.setheading(pi/2)
        t.down()
        for _ in range(2):
            t.forward(2*self.x_lim + 4)
            t.right(pi/2)
            t.forward(2*self.y_lim + 4)
            t.right(pi/2)
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
            #print(f"Landmark 0: ({x},{y})")
            self.landmarks[0].draw(self.wn)
            for i in range(1, self.nb_landmarks):
                while True:
                    x = rd.uniform(-lim_x, lim_x)
                    y = rd.uniform(-lim_y, lim_y)
                    min_dist = 1000
                    for landmark in self.landmarks:
                        dist = sqrt((x-landmark.x)**2 + (y-landmark.y)**2)
                        if dist < min_dist:
                            min_dist = dist
                    # Enforce a minimum space between landmarks
                    if min_dist > min_spacing:
                        break
                self.landmarks += [Landmark(x,y, i)]    
                self.landmarks[-1].draw(self.wn)
                #print(f"Landmark {i}: ({x},{y})")
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
        self.x = distance * cos(angle)
        self.y = distance * sin(angle)
        
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
        self.theta_sigma = 0.1
        self.step_sigma = 0.2
        self.x, self.y = self.pos()
        self.theta = self.heading()
        
    def set(self, x, y, theta):
         self.setposition(x, y)
         self.x, self.y = x, y
         self.setheading(theta)
         self.theta = theta
         
    def move(self, step, theta):
        #self.right(theta)  
        #self.forward(step)
        self.x, self.y = self.pos()
        theta = self.heading() + theta
        x = self.x + step * np.cos(theta)
        y = self.y + step * np.sin(theta)
        self.x, self.y, self.theta = x, y, theta
        self.goto(self.x, self.y)
        self.setheading(self.theta)


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
        #print(f"Robot x: {x}, y: {y}, theta: {theta}")
        for Id, landmark in enumerate(landmarks):
            #print(f"Landmark ID: {Id}, x: {landmark.x}, y: {landmark.y}")
            dx = landmark.x - x
            dy = landmark.y - y
            distance = sqrt(dx*dx + dy*dy)
            if distance < self.max_meas_range:
                angle = atan2(dy, dx) - theta
                #print(f"angle: {angle}, theta: {theta}")
                while angle >= 2*pi:
                    angle -= 2*pi
                while angle <= -2*pi:
                    angle += 2*pi
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
        self.step_sigma = 0.8 #1.1
        self.theta_sigma = 0.2
        
        self.weight = 0.0
        # Standard deviation for weighting
        self.distance_sigma = 20
        # distance_sigma^2 = sigma_x^2 + .sigma_y^2
        self.sigma_x = 14
        self.sigma_y = 14
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

    def gaussian(self, mu, sigma, x):
        return np.exp(-((mu-x)**2)/(2*sigma**2)) / (sqrt(2*pi)*sigma)

    def associate_landmarks(self, robot_measurements):
        """
        Take the robot measurement for x and y and transform to map coordinate
        supposing that the robot is at the particle position.
        Then associate the landmark that is the closest to the map coordinates.
        """
        theta = self.theta
        associations = []
        for meas in robot_measurements:
            #print(f"meas.x: {meas.x}, meas.y: {meas.y}, meas.Id: {meas.landmark_id}")
            x_map = self.x + (cos(theta) * meas.x) - (sin(theta) * meas.y);
            y_map = self.y + (sin(theta) * meas.x) + (cos(theta) * meas.y);
            closest = None
            min_dist = 1000
            for landmark in self.floorplan.landmarks:
                dist = sqrt((landmark.x - x_map)**2 + (landmark.y - y_map)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest = landmark
            #print(f"x_map: {x_map}, y_map: {y_map}, closest: {closest.x}, {closest.y}")
            associations.append((x_map, y_map, closest))  
        return associations

    def update_weight_0(self, robot_measurements):
        self.weight = 1.0
        for x_map, y_map, landmark in self.associate_landmarks(robot_measurements):
            self.weight *= self.gaussian(landmark.x, self.sigma_x, x_map)
            self.weight *= self.gaussian(landmark.y, self.sigma_y, y_map)

    
    def update_weight_1(self, robot_measurements):
        self.weight = 0.0
        
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
                if diff_angle > pi:
                    diff_angle = diff_angle - 2*pi
                distance_weight = self.gaussian(robot_dist, self.distance_sigma,
                                                particle_dist)
                #print(f"distance weight: {distance_weight}")
                angle_weight = self.gaussian(0, self.angle_sigma, diff_angle)
                #print(f"angle weight: {angle_weight}")
                weight = distance_weight * angle_weight
                if weight > best_weight:
                    best_weight = weight
                    selected_landmark = meas.landmark_id
            #print(f"\tParticle meas landmark {selected_landmark}")        
            self.weight += best_weight
 

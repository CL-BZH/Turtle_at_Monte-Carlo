#!/usr/bin/env python3

from mc_turtle import *

#Landmark 0: (121.68234460415977,34.197601273212484)
#Landmark 1: (-52.99160487358641,-124.4965354750137)
#Landmark 2: (-24.371889641058104,-0.36786214345718804)
#Landmark 3: (-125.91402231540933,19.61204475881749)

# Example of landmarks positions
landmarks = [Landmark(-39.9, 1.64, 0, '#ffd700')]
landmarks += [Landmark(59.3, -39.8, 1, '#ffa500')]
landmarks += [Landmark(62.3, 117.1, 2, '#8a2be2')]
landmarks += [Landmark(-58.2, 103.15, 3, '#ff7f50')]

        
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
        xi = sqrt(si * x / y)
        yi = xi * y / x
        x0 = rd.uniform(-self.floorplan.x_lim, -self.floorplan.x_lim + xi)
        y0 = rd.uniform(-self.floorplan.y_lim, -self.floorplan.y_lim + yi)
        particles_count = 0
        for i in range(int(round((x/xi)))): 
            for j in range(int(round((y/yi)))):
                particle = Particle(self.floorplan)
                pos_x = x0 + i * xi
                pos_y = y0 + j * yi
                theta = rd.uniform(0, pi * 2)
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
        #x, y = self.robot.pos()
        heading = self.robot.heading()
        if with_error == True:
            self.robot.move_with_error(forward_step, move_angle)
        else:
            self.robot.move(forward_step, move_angle)
        # Update particles positions according to the movement that
        # the robot was supposed to do (if there would be no movement error)
        for particle in self.particles:
            particle.predict(forward_step, move_angle)
            
        self.floorplan.wn.tracer(1)

    def measure(self, trace=False):
        self.floorplan.wn.tracer(0)
        self.robot.measure(self.floorplan.landmarks, trace)
        self.floorplan.wn.tracer(1)

    def updates_particles_weights(self, update_method=0):
        self.floorplan.wn.tracer(0)
        for particle in self.particles:
            particle.measure(self.floorplan.landmarks, False)
            if update_method == 0:
                particle.update_weight_0(self.robot.measurements)
            else:
                particle.update_weight_1(self.robot.measurements)
        self.floorplan.wn.tracer(1)
        
    def resample_particles(self, update_method=0):
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
            if update_method == 0:
                weight *= 10**12
            if weight > max_weight:
                max_weight = weight
            total_weight += weight
            cumulative_weights.append(total_weight)

        if True:
            print(f"Resampling count: {self.resampling_count}, Max weight: {max_weight}")
            if max_weight < 0.02:
                self.resampling_count += 1
                if self.resampling_count > 7:
                    print(f"+Spread particles+")
                    self.spread_particles()
                    self.resampling_count = 0
                    return
            elif max_weight < 0.035:
                self.resampling_count += 1
                if self.resampling_count > 14:
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
        angle = pi/60
        while True:
            #self.measure(True)
            x, y = self.robot.pos()
            if (self.floorplan.is_within_bounderies(x, y) == False):
                # This is an artefac to keep the robot within some bounderies
                self.move(step+1, angle+pi, self.with_error)
            else:    
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



def main(args=None):
    # Set the scene
    screen_width = 450
    screen_height = 450
    x_lim = 160
    y_lim = 160
    nb_landmarks = 5
    
    min_dist_between_landmarks = 100
    
    floorplan = Floorplan(screen_width, screen_height,
                          x_lim, y_lim, nb_landmarks)
    floorplan.set_screen("Tracking the turtle")
    floorplan.draw_bounderies()
    #floorplan.draw_landmarks(min_dist_between_landmarks, landmarks)
    floorplan.draw_landmarks(min_dist_between_landmarks)
    
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

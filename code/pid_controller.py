# Copyright (C) 2017 Electric Movement Inc.
# All Rights Reserved.

# Author: Brandon Kinman


class PIDController:
    def __init__(self, kp = 0.0, ki = 0.0, kd = 0.0, max_windup = 10):
        # pid parameters
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        # relevant parameters
        self.last_timestamp = 0.0
        self.set_point      = 0.0
        self.start_time     = 0.0
        self.error_sum      = 0.0
        self.last_error     = 0.0
        self.max_windup     = float(max_windup)

        # control history
        self.u_p = [0]
        self.u_i = [0]
        self.u_d = [0]
        
    def reset(self):
        # relevant parameters
        self.last_timestamp = 0.0
        self.set_point      = 0.0
        self.start_time     = 0.0
        self.error_sum      = 0.0
        self.last_error     = 0.0              

        # control history
        self.u_p = [0]
        self.u_i = [0]
        self.u_d = [0]

    def setTarget(self, target):
        self.set_point = float(target)        

    def setKP(self, kp):
        self.kp = float(kp)        

    def setKI(self, ki):
        self.ki = float(ki)

    def setKD(self, kd):
        self.kd = float(kd)

    def setMaxWindup(self, max_windup):
        self.max_windup     = float(max_windup)

    def update(self, measured_value, timestamp):
        dt = timestamp - self.last_timestamp
        if dt == 0:
            return 0 # dt is 0, no time has passed

        # Calculate error
        error = self.set_point - measured_value

        # Reset last timestamp
        self.last_timestamp = timestamp

        # Calculate error sum for integral term
        self.error_sum += error

        # Calculate delta error for derivative term
        delta_error = error - self.last_error

        # Reset/update last error with present error
        self.last_error = error

        # Calculate p, i, d errors
        p = self.kp * error
        i = self.ki * self.error_sum * dt
        d = self.kd * delta_error / dt

        # Calculate control (output) effort
        u = p + i + d

        # Store control effort history
        self.u_p.append(p)
        self.u_i.append(i)
        self.u_d.append(d)

        return u



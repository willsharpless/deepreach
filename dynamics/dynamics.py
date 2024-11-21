from abc import ABC, abstractmethod
from utils import diff_operators, quaternion

import math
import torch

# during training, states will be sampled uniformly by each state dimension from the model-unit -1 to 1 range (for training stability),
# which may or may not correspond to proper test ranges
# note that coord refers to [time, *state], and input refers to whatever is fed directly to the model (often [time, *state, params])
# in the future, code will need to be fixed to correctly handle parameterized models
class Dynamics(ABC):
    def __init__(self, name: str,
    loss_type:str, set_mode:str, 
    state_dim:int, input_dim:int, 
    control_dim:int, disturbance_dim:int, 
    state_mean:list, state_var:list, 
    value_mean:float, value_var:float, value_normto:float, 
    deepreach_model:str):
        self.name = name
        self.loss_type = loss_type
        self.set_mode = set_mode
        self.state_dim = state_dim 
        self.input_dim = input_dim
        self.control_dim = control_dim
        self.disturbance_dim = disturbance_dim
        self.state_mean = torch.tensor(state_mean) 
        self.state_var = torch.tensor(state_var)
        self.value_mean = value_mean
        self.value_var = value_var
        self.value_normto = value_normto
        self.deepreach_model = deepreach_model
        assert self.loss_type in ['brt_hjivi', 'brat_hjivi'], f'loss type {self.loss_type} not recognized'
        if self.loss_type == 'brat_hjivi':
            assert callable(self.reach_fn) and callable(self.avoid_fn)
        assert self.set_mode in ['reach', 'avoid'], f'set mode {self.set_mode} not recognized'
        for state_descriptor in [self.state_mean, self.state_var]:
            assert len(state_descriptor) == self.state_dim, 'state descriptor dimension does not equal state dimension, ' + str(len(state_descriptor)) + ' != ' + str(self.state_dim)
    
    # ALL METHODS ARE BATCH COMPATIBLE

    # MODEL-UNIT CONVERSIONS (TODO: refactor into separate model-unit conversion class?)

    # convert model input to real coord
    def input_to_coord(self, input):
        coord = input.clone()
        coord[..., 1:] = (input[..., 1:] * self.state_var.to(device=input.device)) + self.state_mean.to(device=input.device)
        return coord

    # convert real coord to model input
    def coord_to_input(self, coord):
        input = coord.clone()
        input[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)) / self.state_var.to(device=coord.device)
        return input

    # convert model io to real value
    def io_to_value(self, input, output):
        if self.deepreach_model=="diff":
            return (output * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepreach_model=="exact":
            return (output * input[..., 0] * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        else:
            return (output * self.value_var / self.value_normto) + self.value_mean

    # convert model io to real dv
    def io_to_dv(self, input, output):
        dodi = diff_operators.jacobian(output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)

        if self.deepreach_model=="diff":
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]

            dvds_term1 = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        elif self.deepreach_model=="exact":
            dvdt = (self.value_var / self.value_normto) * \
                (input[..., 0]*dodi[..., 0] + output)

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:] * input[..., 0].unsqueeze(-1)
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        else:
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
            dvds = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
        
        return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

    # ALL FOLLOWING METHODS USE REAL UNITS

    @abstractmethod
    def state_test_range(self):
        raise NotImplementedError

    @abstractmethod
    def equivalent_wrapped_state(self, state):
        raise NotImplementedError

    @abstractmethod
    def dsdt(self, state, control, disturbance):
        raise NotImplementedError
    
    @abstractmethod
    def boundary_fn(self, state):
        raise NotImplementedError

    @abstractmethod
    def sample_target_state(self, num_samples):
        raise NotImplementedError

    @abstractmethod
    def cost_fn(self, state_traj):
        raise NotImplementedError

    @abstractmethod
    def hamiltonian(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_control(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def plot_config(self):
        raise NotImplementedError

class ParameterizedVertDrone2D(Dynamics):
    def __init__(self, gravity:float, input_multiplier_max:float, input_magnitude_max:float):
        self.gravity = gravity                             # g
        self.input_multiplier_max = input_multiplier_max   # k_max
        self.input_magnitude_max = input_magnitude_max     # u_max
        super().__init__(
            name="ParameterizedVertDrone2D",loss_type='brt_hjivi', set_mode='avoid',
            state_dim=3, input_dim=4, control_dim=1, disturbance_dim=0,
            state_mean=[0, 1.5, self.input_multiplier_max/2], # v, z, k
            state_var=[4, 2, self.input_multiplier_max/2],    # v, z, k
            value_mean=0.25,
            value_var=0.5,
            value_normto=0.02,
            deepreach_model="exact",
        )

    def state_test_range(self):
        return [
            [-4, 4],                        # v
            [-0.5, 3.5],                    # z
            [0, self.input_multiplier_max], # k
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    # ParameterizedVertDrone2D dynamics
    # \dot v = k*u - g
    # \dot z = v
    # \dot k = 0
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 2]*control[..., 0] - self.gravity
        dsdt[..., 1] = state[..., 0]
        dsdt[..., 2] = 0
        return dsdt

    def boundary_fn(self, state):
        return -torch.abs(state[..., 1] - 1.5) + 1.5

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        raise NotImplementedError

    def hamiltonian(self, state, dvds):
        return state[..., 2]*torch.abs(dvds[..., 0]*self.input_magnitude_max) \
                - dvds[..., 0]*self.gravity \
                + dvds[..., 1]*state[..., 0]
    
    def optimal_control(self, state, dvds):
        raise NotImplementedError

    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError

    def plot_config(self):
        return {
            'state_slices': [0, 1.5, self.input_multiplier_max/2],
            'state_labels': ['v', 'z', 'k'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

class Air3D(Dynamics):
    def __init__(self, collisionR:float, velocity:float, omega_max:float, angle_alpha_factor:float):
        self.collisionR = collisionR
        self.velocity = velocity
        self.omega_max = omega_max
        self.angle_alpha_factor = angle_alpha_factor
        super().__init__(
            name="Air3D",loss_type='brt_hjivi', set_mode='avoid',
            state_dim=3, input_dim=4, control_dim=1, disturbance_dim=1,
            state_mean=[0, 0, 0], 
            state_var=[1, 1, self.angle_alpha_factor*math.pi],
            value_mean=0.25, 
            value_var=0.5, 
            value_normto=0.02,
            deepreach_model="exact",
        )

    def state_test_range(self):
        return [
            [-1, 1],
            [-1, 1],
            [-math.pi, math.pi],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state

    # Air3D dynamics
    # \dot x    = -v + v \cos \psi + u y
    # \dot y    = v \sin \psi - u x
    # \dot \psi = d - u
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = -self.velocity + self.velocity*torch.cos(state[..., 2]) + control[..., 0]*state[..., 1]
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 2]) - control[..., 0]*state[..., 0]
        dsdt[..., 2] = disturbance[..., 0] - control[..., 0]
        return dsdt
    
    def boundary_fn(self, state):
        return torch.norm(state[..., :2], dim=-1) - self.collisionR

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, dvds):
        ham = self.omega_max * torch.abs(dvds[..., 0] * state[..., 1] - dvds[..., 1] * state[..., 0] - dvds[..., 2])  # Control component
        ham = ham - self.omega_max * torch.abs(dvds[..., 2])  # Disturbance component
        ham = ham + (self.velocity * (torch.cos(state[..., 2]) - 1.0) * dvds[..., 0]) + (self.velocity * torch.sin(state[..., 2]) * dvds[..., 1])  # Constant component
        return ham

    def optimal_control(self, state, dvds):
        det = dvds[..., 0]*state[..., 1] - dvds[..., 1]*state[..., 0]-dvds[..., 2]
        return (self.omega_max * torch.sign(det))[..., None]
    
    def optimal_disturbance(self, state, dvds):
        return (-self.omega_max * torch.sign(dvds[..., 2]))[..., None]
    
    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x', 'y', 'theta'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

class Dubins3D(Dynamics):
    def __init__(self, goalR:float, velocity:float, omega_max:float, angle_alpha_factor:float, set_mode:str, freeze_model:bool):
        self.goalR = goalR
        self.velocity = velocity
        self.omega_max = omega_max
        self.angle_alpha_factor = angle_alpha_factor
        self.freeze_model = freeze_model
        super().__init__(
            name="Dubins3D", loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=3, input_dim=4, control_dim=1, disturbance_dim=0,
            state_mean=[0, 0, 0], 
            state_var=[1, 1, self.angle_alpha_factor*math.pi],
            value_mean=0.25, 
            value_var=0.5, 
            value_normto=0.02,
            deepreach_model="exact"
        )

    def state_test_range(self):
        return [
            [-1, 1],
            [-1, 1],
            [-math.pi, math.pi],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state
        
    # Dubins3D dynamics
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control, disturbance):
        if self.freeze_model:
            raise NotImplementedError
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity*torch.cos(state[..., 2])
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 2])
        dsdt[..., 2] = control[..., 0]
        return dsdt
    
    def boundary_fn(self, state):
        return torch.norm(state[..., :2], dim=-1) - self.goalR

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, dvds):
        if self.freeze_model:
            raise NotImplementedError
        if self.set_mode == 'reach':
            return self.velocity*(torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) - self.omega_max * torch.abs(dvds[..., 2]) 
        elif self.set_mode == 'avoid':
            return self.velocity*(torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) + self.omega_max * torch.abs(dvds[..., 2])

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            return (-self.omega_max*torch.sign(dvds[..., 2]))[..., None]
        elif self.set_mode == 'avoid':
            return (self.omega_max*torch.sign(dvds[..., 2]))[..., None]

    def optimal_disturbance(self, state, dvds):
        return 0
    
    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x', 'y', r'$\theta$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

class Linear2D(Dynamics):
    def __init__(self):
        # __init__(self, goalR:float, u_max:float, d_max:float, A:tensor, B:tensor, C:tensor, set_mode:str) #FIXME
        goalR, u_max, d_max, set_mode = 0.25, 0.5, 0.3, "reach" 
        self.a11, self.a12, self.a21, self.a22 = 0., .5, -1., -1. # FIXME lin algebra will be faster and cleaner
        self.b1, self.b2, self.c1, self.c2 = .4, .1, 0., .1

        self.goalR = goalR
        self.u_max, self.d_max = u_max, d_max
        super().__init__(
            name = "linear2D", loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=2, input_dim=3, control_dim=2, disturbance_dim=2, # TODO What is input_dim and what should it be?
            state_mean=[0, 0], 
            state_var=[1, 1],
            value_mean=0.25, 
            value_var=0.5, 
            value_normto=0.02,
            deepreach_model="exact",
        )

    def state_test_range(self):
        return [
            [-1, 1],
            [-1, 1],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state
        
    # Linear dynamics
    # \dot x    = a11 x + a12 y + b1 * u1 + c1 * d1
    # \dot y    = a21 x + a22 y + b2 * u2 + c2 * d2
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.a11 * state[..., 0] + self.a12 * state[..., 1] + self.b1 * control[..., 0] + self.c1 * disturbance[..., 0]
        dsdt[..., 1] = self.a21 * state[..., 0] + self.a22 * state[..., 1] + self.b2 * control[..., 1] + self.c2 * disturbance[..., 1]
        return dsdt
    
    def boundary_fn(self, state):
        # return torch.norm(state[..., :2], dim=-1) - self.goalR
        return 0.5 * (torch.square(torch.norm(state[..., :2], dim=-1)) - torch.square(self.goalR))

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, dvds):
        pAx = dvds[..., 0] * (self.a11 * state[..., 0] + self.a12 * state[..., 1]) + dvds[..., 1] * (self.a21 * state[..., 0] + self.a22 * state[..., 1])
        pb = self.b1 * torch.abs(dvds[..., 0]) + self.b2 * torch.abs(dvds[..., 1])
        pc = self.c1 * torch.abs(dvds[..., 0]) + self.c2 * torch.abs(dvds[..., 1])
        if self.set_mode == 'reach':
            return pAx - self.u_max * pb + self.d_max * pc
        elif self.set_mode == 'avoid':
            return pAx + self.u_max * pb - self.d_max * pc
        # if self.set_mode == 'reach':
        #     return self.velocity*(torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) - self.omega_max * torch.abs(dvds[..., 2]) 
        # elif self.set_mode == 'avoid':
        #     return self.velocity*(torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) + self.omega_max * torch.abs(dvds[..., 2])

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            return torch.cat((-self.u_max * torch.sign(dvds[..., 0]), -self.u_max * torch.sign(dvds[..., 1])), dim=-1)
        elif self.set_mode == 'avoid':
            return torch.cat((self.u_max * torch.sign(dvds[..., 0]), self.u_max * torch.sign(dvds[..., 1])), dim=-1)
        # if self.set_mode == 'reach':
        #     return (-self.omega_max*torch.sign(dvds[..., 2]))[..., None]
        # elif self.set_mode == 'avoid':
        #     return (self.omega_max*torch.sign(dvds[..., 2]))[..., None]

    def optimal_disturbance(self, state, dvds):
        if self.set_mode == 'reach':
            return torch.cat((self.d_max * torch.sign(dvds[..., 0]), self.d_max * torch.sign(dvds[..., 1])), dim=-1)
        elif self.set_mode == 'avoid':
            return torch.cat((-self.d_max * torch.sign(dvds[..., 0]), -self.d_max * torch.sign(dvds[..., 1])), dim=-1)
    
    def plot_config(self):
        return {
            'state_slices': [0, 0],
            'state_labels': ['x', 'y'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

class LessLinear2D(Dynamics):
    # def __init__(self, gamma:float, mu:float, alpha:float):
    def __init__(self):
        # __init__(self, goalR:float, u_max:float, d_max:float, A:tensor, B:tensor, C:tensor, set_mode:str) #FIXME
        gamma, mu, alpha = 20, -20, 1
        goalR, u_max, d_max, set_mode = 0.25, 0.5, 0.3, "reach" 
        self.a11, self.a12, self.a21, self.a22 = 0., .5, -1., -1. # FIXME lin algebra will be faster and cleaner
        self.b1, self.b2, self.c1, self.c2 = .4, .1, 0., .1
        self.gamma, self.mu, self.alpha = gamma, mu, alpha

        self.goalR = goalR
        self.u_max, self.d_max = u_max, d_max
        super().__init__(
            name= "LessLinear2D", loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=2, input_dim=3, control_dim=2, disturbance_dim=2, # TODO What is input_dim and what should it be?
            state_mean=[0, 0], 
            state_var=[1, 1],
            value_mean=0.25, 
            value_var=0.5, 
            value_normto=0.02,
            deepreach_model="exact",
        )

    def state_test_range(self):
        return [
            [-1, 1],
            [-1, 1],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state
        
    # LessLinear dynamics
    # \dot x    = a11 x + a12 y + b1 * u1 + c1 * d1 + mu * sin(alpha * x1) * x2^2
    # \dot y    = a21 x + a22 y + b2 * u2 + c2 * d2 - gamma * x2 * x1^2
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        nl_term =  - self.gamma * state[..., 1] * state[..., 0] * state[..., 0]
        nl_term2 =  self.mu * torch.sin(self.alpha * state[..., 0]) * state[..., 1] * state[..., 1]
        dsdt[..., 0] = self.a11 * state[..., 0] + self.a12 * state[..., 1] + self.b1 * control[..., 0] + self.c1 * disturbance[..., 0] + nl_term2
        dsdt[..., 1] = self.a21 * state[..., 0] + self.a22 * state[..., 1] + self.b2 * control[..., 1] + self.c2 * disturbance[..., 1] + nl_term
        return dsdt
    
    def boundary_fn(self, state):
        # return torch.norm(state[..., :2], dim=-1) - self.goalR
        return 0.5 * (torch.square(torch.norm(state[..., :2], dim=-1)) - self.goalR ** 2)

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, dvds):
        nl_term =  - self.gamma * state[..., 1] * state[..., 0] * state[..., 0]
        nl_term2 =  self.mu * torch.sin(self.alpha * state[..., 0]) * state[..., 1] * state[..., 1]
        pAx = dvds[..., 0] * (self.a11 * state[..., 0] + self.a12 * state[..., 1] + nl_term2) + dvds[..., 1] * (self.a21 * state[..., 0] + self.a22 * state[..., 1] + nl_term)
        pb = self.b1 * torch.abs(dvds[..., 0]) + self.b2 * torch.abs(dvds[..., 1])
        pc = self.c1 * torch.abs(dvds[..., 0]) + self.c2 * torch.abs(dvds[..., 1])
        if self.set_mode == 'reach':
            return pAx - self.u_max * pb + self.d_max * pc
        elif self.set_mode == 'avoid':
            return pAx + self.u_max * pb - self.d_max * pc

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            return torch.cat((-self.u_max * torch.sign(dvds[..., 0]), -self.u_max * torch.sign(dvds[..., 1])), dim=-1)
        elif self.set_mode == 'avoid':
            return torch.cat((self.u_max * torch.sign(dvds[..., 0]), self.u_max * torch.sign(dvds[..., 1])), dim=-1)

    def optimal_disturbance(self, state, dvds):
        if self.set_mode == 'reach':
            return torch.cat((self.d_max * torch.sign(dvds[..., 0]), self.d_max * torch.sign(dvds[..., 1])), dim=-1)
        elif self.set_mode == 'avoid':
            return torch.cat((-self.d_max * torch.sign(dvds[..., 0]), -self.d_max * torch.sign(dvds[..., 1])), dim=-1)
    
    def plot_config(self):
        return {
            'state_slices': [0, 0],
            'state_labels': ['x', 'y'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

class LessLinearND(Dynamics):
    def __init__(self, N:int, gamma:float, mu:float, alpha:float):
    # def __init__(self, N:int):
        # gamma, mu, alpha = 0, 0, 0 
        # gamma, mu, alpha = 20, 0, 0
        # gamma, mu, alpha = 20, -20, 1
        goalR, u_max, d_max, set_mode = 0.15, 0.5, 0.3, "reach" # TODO: unfix

        self.N = N 
        self.u_max, self.d_max = u_max, d_max
        self.input_center = torch.zeros(N-1)
        self.input_shape = "box"
        self.game = set_mode
        
        self.A = (-0.5 * torch.eye(N) - torch.cat((torch.cat((torch.zeros(1,1),torch.ones(N-1,1)),0),torch.zeros(N,N-1)),1)).cuda()
        self.B = torch.cat((torch.zeros(1,N-1), 0.4*torch.eye(N-1)), 0)
        self.Bumax = u_max * torch.matmul(self.B, torch.ones(self.N-1)).unsqueeze(0).unsqueeze(0).cuda()
        self.C = torch.cat((torch.zeros(1,N-1), 0.1*torch.eye(N-1)), 0)
        self.Cdmax = d_max * torch.matmul(self.C, torch.ones(self.N-1)).unsqueeze(0).unsqueeze(0).cuda()
        self.gamma, self.mu, self.alpha = gamma, mu, alpha
        self.gamma_orig, self.mu_orig, self.alpha_orig = gamma, mu, alpha

        self.goalR_2d = goalR
        self.goalR = ((N-1) ** 0.5) * self.goalR_2d # accounts for N-dimensional combination
        self.ellipse_params = torch.cat((((N-1) ** 0.5) * torch.ones(1), torch.ones(N-1) / 1.), 0) # accounts for N-dimensional combination

        self.u_max, self.d_max = u_max, d_max
        super().__init__(
            loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=N, input_dim=N+1, control_dim=N-1, disturbance_dim=N-1, # TODO What is input_dim and what should it be?
            state_mean=[0 for _ in range(N)], 
            state_var=[1 for _ in range(N)],
            value_mean=0.25, 
            value_var=0.5, 
            value_normto=0.02,
            deepreach_model="exact",
        )

    def vary_nonlinearity(self, epsilon):
        self.gamma = epsilon * self.gamma_orig
        self.mu = epsilon * self.mu_orig
        self.alpha = epsilon * self.alpha_orig

    def state_test_range(self):
        return [[-1, 1] for _ in range(self.N)]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state
        
    # LessLinear dynamics
    # \dot xN    = (aN \cdot x) + (no ctrl or dist) + mu * sin(alpha * xN) * xN^2
    # \dot xi    = (ai \cdot x) + bi * ui + ci * di - gamma * xi * xN^2
    # i.e.
    # \dot x = Ax + Bu + Cd + NLterm(x, gamma, mu, alpha)
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        # nl_term =  - self.gamma * state[..., 1] * state[..., 0] * state[..., 0]
        # nl_term2 =  self.mu * torch.sin(self.alpha * state[..., 0]) * state[..., 1] * state[..., 1]
        # dsdt[..., 0] = self.a11 * state[..., 0] + self.a12 * state[..., 1] + self.b1 * control[..., 0] + self.c1 * disturbance[..., 0] + nl_term2
        # dsdt[..., 1] = self.a21 * state[..., 0] + self.a22 * state[..., 1] + self.b2 * control[..., 1] + self.c2 * disturbance[..., 1] + nl_term
        nl_term_N = self.mu * torch.sin(self.alpha * state[..., 0]) * state[..., 0] * state[..., 0]
        nl_term_i = torch.multiply(-self.gamma * state[..., 0] * state[..., 0], state[..., 1:])
        dsdt[..., :] = torch.matmul(self.A, state[..., :]) + torch.matmul(self.B, control[..., :]) + torch.matmul(self.C, disturbance[..., :]) + torch.cat((nl_term_N, nl_term_i), 0)
        return dsdt
    
    def boundary_fn(self, state):
        if self.ellipse_params.device != state.device: # FIXME: Patch to cover de/attached state bug
            if state.device.type == 'cuda':
                self.ellipse_params = self.ellipse_params.cuda()
            else:
                self.ellipse_params = self.ellipse_params.cpu()
        return 0.5 * (torch.square(torch.norm(self.ellipse_params * state[..., :], dim=-1)) - (self.goalR ** 2))
        # return 0.5 * (torch.square(torch.norm(torch.cat((((self.N-1)**0.5)*torch.ones(1),torch.ones(self.N-1)),0) * state[..., :], dim=-1)) - (self.goalR ** 2))

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, dvds):

        nl_term_N = (self.mu * torch.sin(self.alpha * state[..., 0]) * state[..., 0] * state[..., 0]).unsqueeze(-1)
        nl_term_i = (-self.gamma * state[..., 0] * state[..., 0]).t() * state[..., 1:]
        pAx = (dvds * (torch.matmul(state, self.A.t()) + torch.cat((nl_term_N, nl_term_i), 2))).sum(2)
        pBumax = (torch.abs(dvds) * self.Bumax).sum(2)
        pCdmax = (torch.abs(dvds) * self.Cdmax).sum(2)

        if self.set_mode == 'reach':
            return pAx - pBumax + pCdmax
        elif self.set_mode == 'avoid':
            return pAx + pBumax - pCdmax

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            # return torch.cat((-self.u_max * torch.sign(dvds[..., 0]), -self.u_max * torch.sign(dvds[..., 1])), dim=-1)
            return -self.u_max * torch.sign(dvds[..., :])
        elif self.set_mode == 'avoid':
            # return torch.cat((self.u_max * torch.sign(dvds[..., 0]), self.u_max * torch.sign(dvds[..., 1])), dim=-1)
            return self.u_max * torch.sign(dvds[..., :])

    def optimal_disturbance(self, state, dvds):
        if self.set_mode == 'reach':
            # return torch.cat((self.d_max * torch.sign(dvds[..., 0]), self.d_max * torch.sign(dvds[..., 1])), dim=-1)
            return self.d_max * torch.sign(dvds[..., :])
        elif self.set_mode == 'avoid':
            # return torch.cat((-self.d_max * torch.sign(dvds[..., 0]), -self.d_max * torch.sign(dvds[..., 1])), dim=-1)
            return -self.d_max * torch.sign(dvds[..., :])
    
    def plot_config(self): # FIXME
        return {
            'state_slices': [0 for _ in range(self.N)],
            'state_labels': ['xN'] + ['x' + str(i) for i in range(1, self.N)],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }
    
# class LinearND(LessLinearND):
#     def __init__(self, N:int, gamma:float, mu:float, alpha:float):
#         super().__init__(self, N, 0, 0, 0)

class QuadrotorEuler(Dynamics):
    def __init__(self, collective_thrust_max: float,  set_mode: str):  # simpler quadrotor
        self.collective_thrust_max = collective_thrust_max
        # self.body_rate_acc_max = body_rate_acc_max
        self.m = 0.034  # mass
        self.arm_l = 0.04
        self.CT = 1
        self.CM = 0.016
        self.Gz = -9.8

        self.dwx_max = 10
        self.dwy_max = 10
        self.dwz_max = 5

        # self.collisionR = collisionR
        self.cylinder1_info = [0, 0, 0.4]  # x y r
        self.cylinder2_info = [0, 0.5, 0.4]  # x y r
        self.sphere1_info = [0, -1, 0, 0.5]  # x y z r
        self.ground = -2
        self.ceiling = 2

        super().__init__(
            name="QuadrotorEuler", loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=12, input_dim=13, nn_input_dim=14, control_dim=4, disturbance_dim=0,
            state_mean=[0 for i in range(12)],
            state_var=[3, 3, 3, math.pi, math.pi, math.pi, 5, 5, 5, 15, 15, 5],
            value_mean=(math.sqrt(3**2 + 3**2)) / 2,
            value_var=math.sqrt(3**2 + 3**2),
            value_normto=0.02,
            deepReach_model='exact', method_='vanilla', quaternion_start_dim=-1
        )

    def control_range(self, state):
        return [[-self.collective_thrust_max, self.collective_thrust_max],
                [-self.dwx_max, self.dwx_max],
                [-self.dwy_max, self.dwy_max],
                [-self.dwz_max, self.dwz_max]]

    def state_test_range(self):
        return [
            [-3, 3],
            [-3, 3],
            [-3, 3],
            [-math.pi, math.pi],
            [-math.pi, math.pi],
            [-math.pi, math.pi],
            [-1, 1],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
        ]

    def periodic_transform_fn(self, input):
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1]+1
        transformed_input = torch.zeros(output_shape)
        transformed_input[..., :3] = input[..., :3]
        transformed_input[..., 3:7] = quaternion.get_quaternion_from_euler(
            input[..., 3], input[..., 4], input[..., 5])
        transformed_input[..., 7:] = input[..., 6:]
        return transformed_input.cuda()

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 3] = (
            wrapped_state[..., 3] + math.pi) % (2 * math.pi) - math.pi

        wrapped_state[..., 4] = (
            wrapped_state[..., 4] + math.pi) % (2 * math.pi) - math.pi

        wrapped_state[..., 5] = (
            wrapped_state[..., 5] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state

    def dsdt(self, state, control, disturbance):
        phi = state[..., 3] * 1.0  # x
        theta = state[..., 4] * 1.0  # y
        psi = state[..., 5] * 1.0  # z
        vx = state[..., 6] * 1.0
        vy = state[..., 7] * 1.0
        vz = state[..., 8] * 1.0
        wx = state[..., 9] * 1.0
        wy = state[..., 10] * 1.0
        wz = state[..., 11] * 1.0
        f = control[..., 0] * 1.0
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = torch.cos(theta)*torch.cos(psi)*vx + (torch.sin(theta)*torch.sin(phi)*torch.cos(psi) - torch.cos(
            phi)*torch.sin(psi))*vy + (torch.sin(theta)*torch.cos(phi)*torch.cos(psi) + torch.sin(phi)*torch.sin(psi))*vz
        dsdt[..., 1] = -(torch.cos(theta)*torch.sin(psi)*vx + (torch.sin(theta)*torch.sin(phi)*torch.sin(psi) + torch.cos(
            phi)*torch.cos(psi))*vy + (torch.sin(theta)*torch.cos(phi)*torch.sin(psi) - torch.sin(phi)*torch.cos(psi))*vz)
        dsdt[..., 2] = torch.sin(theta)*vx - torch.cos(theta) * \
            torch.sin(phi)*vy - torch.cos(theta)*torch.cos(phi)*vz
        dsdt[..., 3] = wx+torch.sin(phi)*torch.tan(theta) * \
            wy+torch.cos(phi)*torch.tan(theta)*wz
        dsdt[..., 4] = torch.cos(phi)*wy-torch.sin(phi)*wz
        dsdt[..., 5] = (torch.sin(phi)*wy+torch.cos(phi)*wz)/torch.cos(theta)
        dsdt[..., 6] = wz*vy-wy*vz
        dsdt[..., 7] = wx*vz-wz*vx
        dsdt[..., 8] = wy*vx-wx*vy + f/self.m
        dsdt[..., 9] = control[..., 1] * 1.0 - 5 * wy * wz / 9.0
        dsdt[..., 10] = control[..., 2] * 1.0 + 5 * wx * wz / 9.0
        dsdt[..., 11] = control[..., 3] * 1.0

        # here we add the components of G.. (inverse of ZYX euler angles)
        dsdt[..., 6] += torch.sin(-theta)*self.Gz
        dsdt[..., 7] += -torch.cos(-theta)*torch.sin(-phi)*self.Gz
        dsdt[..., 8] += torch.cos(-phi)*torch.cos(-theta)*self.Gz
        return dsdt

    # def dsdt(state, control, disturbance):
    #     '''
    #     state: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz] m, m, m, rad, rad, rad, m/s, m/s, m/s, rad/s, rad/s, rad/s
    #     control: [m1, m2, m3, m4] Newton
    #     '''
    #     phi = state[..., 3] * 1.0  # roll
    #     theta = state[..., 4] * 1.0 # pitch
    #     psi = state[..., 5] * 1.0   # yaw
    #     vx = state[..., 6] * 1.0 # x velocity
    #     vy = state[..., 7] * 1.0 # y velocity
    #     vz = state[..., 8] * 1.0  # z velocity
    #     wx = state[..., 9] * 1.0 # roll rate
    #     wy = state[..., 10] * 1.0 # pitch rate
    #     wz = state[..., 11] * 1.0  # yaw rate

    #     f = control[..., 0]+control[..., 1]+control[..., 2]+control[..., 3] # net thrust
    #     tau_x = (control[..., 0]+control[..., 1]-control[..., 2]-control[..., 3])*arm_l/math.sqrt(2)
    #     tau_y = (-control[..., 0]+control[..., 1]+control[..., 2]-control[..., 3])*arm_l/math.sqrt(2)
    #     tau_z = (control[..., 0]-control[..., 1]+control[..., 2]-control[..., 3])*CM
    #     dsdt = torch.zeros_like(state)

    #     dsdt[..., 0] = cos(theta)*cos(psi)*vx + (sin(theta)*sin(phi)*cos(psi) - cos(
    #         phi)*sin(psi))*vy + (sin(theta)*cos(phi)*cos(psi) + sin(phi)*sin(psi))*vz
    #     dsdt[..., 1] = cos(theta)*sin(psi)*vx + (sin(theta)*sin(phi)*sin(psi) + cos(
    #         phi)*cos(psi))*vy + (sin(theta)*cos(phi)*sin(psi) - sin(phi)*cos(psi))*vz
    #     dsdt[..., 2] = -sin(theta)*vx + cos(theta) * \
    #         sin(phi)*vy + cos(theta)*cos(phi)*vz
    #     dsdt[..., 3] = wx+sin(phi)*tan(theta) * \
    #         wy+cos(phi)*tan(theta)*wz
    #     dsdt[..., 4] = cos(phi)*wy-sin(phi)*wz
    #     dsdt[..., 5] = (sin(phi)*wy+cos(phi)*wz)/cos(theta)
    #     dsdt[..., 6] = wz*vy-wy*vz
    #     dsdt[..., 7] = wx*vz-wz*vx
    #     dsdt[..., 8] = wy*vx-wx*vy + f/m
    #     dsdt[..., 9] = (tau_x)/Ixx - (Izz-Iyy)/Ixx* wy * wz
    #     dsdt[..., 10] = (tau_y)/Iyy + (Izz-Ixx)/Iyy* wx * wz
    #     dsdt[..., 11] = (tau_z)/Izz

    #     # here we add the components of G.. (inverse of ZYX euler angles)
    #     dsdt[..., 6] += sin(-theta)*Gz
    #     dsdt[..., 7] += -cos(-theta)*sin(-phi)*Gz
    #     dsdt[..., 8] += cos(-phi)*cos(-theta)*Gz
    #     return dsdt

    def boundary_fn(self, state):
        dist_cylinder_1 = self.dist_from_cylinder(state, self.cylinder1_info)
        return dist_cylinder_1

    def dist_from_cylinder(self, state, ceilinder_info):
        '''for cylinder with full body collision'''
        phi = state[..., 3] * 1.0  # x
        theta = state[..., 4] * 1.0  # y
        psi = state[..., 5] * 1.0  # z

        vx = (torch.sin(theta)*torch.cos(phi) *
              torch.cos(psi) + torch.sin(phi)*torch.sin(psi))*-1
        vy = (torch.sin(theta)*torch.cos(phi) *
              torch.sin(psi) - torch.sin(phi)*torch.cos(psi))*-1
        vz = - torch.cos(theta)*torch.cos(psi)

        # compute vector from center of quadrotor to the center of cylinder
        p = state[..., :2]*1.0
        p[..., 0] -= ceilinder_info[0]
        p[..., 1] -= ceilinder_info[1]
        px = p[..., 0]
        py = p[..., 1]
        # get full body distance
        dist = torch.norm(p[..., :2], dim=-1)
        # return dist- collisionR
        dist -= torch.sqrt((self.arm_l**2*px**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2)
                           + (self.arm_l**2*py**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2))
        return torch.maximum(dist, torch.zeros_like(dist)) - ceilinder_info[2]

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds, dt):
        if self.set_mode == 'reach':
            raise NotImplementedError

        elif self.set_mode == 'avoid':
            control_ = self.optimal_control(state, dvds)
            disturbance_ = self.optimal_disturbance(state, dvds)
            dsdt_ = self.dsdt(state, control_, disturbance_).detach()
            ham = 0.0
            for i in range(self.state_dim):
                ham += dsdt_[..., i]*dvds[..., i]
            return ham

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError
        elif self.set_mode == 'avoid':
            u1 = self.collective_thrust_max * torch.sign(dvds[..., 8])
            u2 = self.dwx_max * torch.sign(dvds[..., 9])
            u3 = self.dwy_max * torch.sign(dvds[..., 10])
            u4 = self.dwz_max * torch.sign(dvds[..., 11])

        return torch.cat((u1[..., None], u2[..., None], u3[..., None], u4[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'state_labels': ['x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 7,
        }


class Dubins4D(Dynamics):
    def __init__(self, bound_mode:str):
        self.vMin = 0.2
        self.vMax = 14.8
        self.collisionR = 1.5
        self.bound_mode = bound_mode
        assert self.bound_mode in ['v1', 'v2']

        xMean = 0
        yMean = 0
        thetaMean = 0
        vMean = 7.5
        aMean = 0
        oMean = 0

        xVar = 10
        yVar = 10
        thetaVar = 1.2*math.pi
        vVar = 7.5
        aVar = 10
        oVar = 3*math.pi if self.bound_mode == 'v1' else 2.0
        
        super().__init__(
            name = "dubins4D", loss_type='brt_hjivi',
            state_dim=14, input_dim=15,  control_dim=2, disturbance_dim=0,
            state_mean=[xMean, yMean, thetaMean, vMean, xMean, yMean, aMean, aMean, oMean, oMean, aMean, aMean, oMean, oMean],
            state_var=[xVar, yVar, thetaVar, vVar, xVar, yVar, aVar, aVar, oVar, oVar, aVar, aVar, oVar, oVar],
            value_mean=13,
            value_var=14,
            value_normto=0.02,
            deepreach_model="exact",
        )

    def state_test_range(self):
        return [
            [-1, 1],
            [-1, 1],
            [-math.pi, math.pi],
            [self.vMin, self.vMax],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state

    def boundary_fn(self, state):
        return torch.norm(state[..., 0:2] - state[..., 4:6], dim=-1) - self.collisionR

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        raise NotImplementedError

    def dsdt(self, state, control, disturbance):
        raise NotImplementedError

    def hamiltonian(self, state, dvds):
        raise NotImplementedError

    def optimal_control(self, state, dvds):
        raise NotImplementedError

    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError

    def plot_config(self):
        raise NotImplementedError

class NarrowPassage(Dynamics):
    def __init__(self, avoid_fn_weight:float, avoid_only:bool):
        self.L = 2.0

        # # Target positions
        self.goalX = [6.0, -6.0]
        self.goalY = [-1.4, 1.4]

        # State bounds
        self.vMin = 0.001
        self.vMax = 6.50
        self.phiMin = -0.3*math.pi + 0.001
        self.phiMax = 0.3*math.pi - 0.001

        # Control bounds
        self.aMin = -4.0
        self.aMax = 2.0
        self.psiMin = -3.0*math.pi
        self.psiMax = 3.0*math.pi

        # Lower and upper curb positions (in the y direction)
        self.curb_positions = [-2.8, 2.8]

        # Stranded car position
        self.stranded_car_pos = [0.0, -1.8]

        self.avoid_fn_weight = avoid_fn_weight

        self.avoid_only = avoid_only

        super().__init__(
            name = "NarrowPassage", loss_type='brt_hjivi' if self.avoid_only else 'brat_hjivi', set_mode='avoid' if self.avoid_only else 'reach',
            state_dim=10, input_dim=11, control_dim=4, disturbance_dim=0,
            # state = [x1, y1, th1, v1, phi1, x2, y2, th2, v2, phi2]
            state_mean=[
                0, 0, 0, 3, 0, 
                0, 0, 0, 3, 0
            ],
            state_var=[
                8.0, 3.8, 1.2*math.pi, 4.0, 1.2*0.3*math.pi, 
                8.0, 3.8, 1.2*math.pi, 4.0, 1.2*0.3*math.pi,
            ],
            value_mean=0.25*8.0,
            value_var=0.5*8.0,
            value_normto=0.02,
            deepreach_model="exact",
        )

    def state_test_range(self):
        return [
            [-8, 8],
            [-3.8, 3.8],
            [-math.pi, math.pi],
            [-1, 7],
            [-0.3*math.pi, 0.3*math.pi],
            [-8, 8],
            [-3.8, 3.8],
            [-math.pi, math.pi],
            [-1, 7],
            [-0.3*math.pi, 0.3*math.pi],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 4] = (wrapped_state[..., 4] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 7] = (wrapped_state[..., 7] + math.pi) % (2*math.pi) - math.pi 
        wrapped_state[..., 9] = (wrapped_state[..., 9] + math.pi) % (2*math.pi) - math.pi 
        return wrapped_state 

    # NarrowPassage dynamics
    # \dot x   = v * cos(th)
    # \dot y   = v * sin(th)
    # \dot th  = v * tan(phi) / L
    # \dot v   = u1
    # \dot phi = u2
    # \dot x   = ...
    # \dot y   = ...
    # \dot th  = ...
    # \dot v   = ...
    # \dot phi = ...
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 3]*torch.cos(state[..., 2])
        dsdt[..., 1] = state[..., 3]*torch.sin(state[..., 2])
        dsdt[..., 2] = state[..., 3]*torch.tan(state[..., 4]) / self.L
        dsdt[..., 3] = control[..., 0]
        dsdt[..., 4] = control[..., 1]
        dsdt[..., 5] = state[..., 8]*torch.cos(state[..., 7])
        dsdt[..., 6] = state[..., 8]*torch.sin(state[..., 7])
        dsdt[..., 7] = state[..., 8]*torch.tan(state[..., 9]) / self.L
        dsdt[..., 8] = control[..., 2]
        dsdt[..., 9] = control[..., 3]
        return dsdt

    def reach_fn(self, state):
        if self.avoid_only:
            raise RuntimeError
        # vehicle 1
        goal_tensor_R1 = torch.tensor([self.goalX[0], self.goalY[0]], device=state.device)
        dist_R1 = torch.norm(state[..., 0:2] - goal_tensor_R1, dim=-1) - self.L
        # vehicle 2
        goal_tensor_R2 = torch.tensor([self.goalX[1], self.goalY[1]], device=state.device)
        dist_R2 = torch.norm(state[..., 5:7] - goal_tensor_R2, dim=-1) - self.L
        return torch.maximum(dist_R1, dist_R2)
    
    def avoid_fn(self, state):
        # distance from lower curb
        dist_lc_R1 = state[..., 1] - self.curb_positions[0] - 0.5*self.L
        dist_lc_R2 = state[..., 6] - self.curb_positions[0] - 0.5*self.L
        dist_lc = torch.minimum(dist_lc_R1, dist_lc_R2)
        
        # distance from upper curb
        dist_uc_R1 = self.curb_positions[1] - state[..., 1] - 0.5*self.L
        dist_uc_R2 = self.curb_positions[1] - state[..., 6] - 0.5*self.L
        dist_uc = torch.minimum(dist_uc_R1, dist_uc_R2)
        
        # distance from the stranded car
        stranded_car_pos = torch.tensor(self.stranded_car_pos, device=state.device)
        dist_stranded_R1 = torch.norm(state[..., 0:2] - stranded_car_pos, dim=-1) - self.L
        dist_stranded_R2 = torch.norm(state[..., 5:7] - stranded_car_pos, dim=-1) - self.L
        dist_stranded = torch.minimum(dist_stranded_R1, dist_stranded_R2)

        # distance between the vehicles themselves
        dist_R1R2 = torch.norm(state[..., 0:2] - state[..., 5:7], dim=-1) - self.L

        return self.avoid_fn_weight * torch.min(torch.min(torch.min(dist_lc, dist_uc), dist_stranded), dist_R1R2)

    def boundary_fn(self, state):
        if self.avoid_only:
            return self.avoid_fn(state)
        else:
            return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):    
        if self.avoid_only:
            return torch.min(self.avoid_fn(state_traj), dim=-1).values
        else:   
            # return min_t max{l(x(t)), max_k_up_to_t{-g(x(k))}}, where l(x) is reach_fn, g(x) is avoid_fn 
            reach_values = self.reach_fn(state_traj)
            avoid_values = self.avoid_fn(state_traj)
            return torch.min(torch.maximum(reach_values, torch.cummax(-avoid_values, dim=-1).values), dim=-1).values

    def hamiltonian(self, state, dvds):
        optimal_control = self.optimal_control(state, dvds)
        return state[..., 3] * torch.cos(state[..., 2]) * dvds[..., 0] + \
               state[..., 3] * torch.sin(state[..., 2]) * dvds[..., 1] + \
               state[..., 3] * torch.tan(state[..., 4]) * dvds[..., 2] / self.L + \
               optimal_control[..., 0] * dvds[..., 3] + \
               optimal_control[..., 1] * dvds[..., 4] + \
               state[..., 8] * torch.cos(state[..., 7]) * dvds[..., 5] + \
               state[..., 8] * torch.sin(state[..., 7]) * dvds[..., 6] + \
               state[..., 8] * torch.tan(state[..., 9]) * dvds[..., 7] / self.L + \
               optimal_control[..., 2] * dvds[..., 8] + \
               optimal_control[..., 3] * dvds[..., 9]

    def optimal_control(self, state, dvds):
        a1_min = self.aMin * (state[..., 3] > self.vMin)
        a1_max = self.aMax * (state[..., 3] < self.vMax)
        psi1_min = self.psiMin * (state[..., 4] > self.phiMin)
        psi1_max = self.psiMax * (state[..., 4] < self.phiMax)
        a2_min = self.aMin * (state[..., 8] > self.vMin)
        a2_max = self.aMax * (state[..., 8] < self.vMax)
        psi2_min = self.psiMin * (state[..., 9] > self.phiMin)
        psi2_max = self.psiMax * (state[..., 9] < self.phiMax)

        if self.avoid_only:
            a1 = torch.where(dvds[..., 3] < 0, a1_min, a1_max)
            psi1 = torch.where(dvds[..., 4] < 0, psi1_min, psi1_max)
            a2 = torch.where(dvds[..., 8] < 0, a2_min, a2_max)
            psi2 = torch.where(dvds[..., 9] < 0, psi2_min, psi2_max)

        else:
            a1 = torch.where(dvds[..., 3] > 0, a1_min, a1_max)
            psi1 = torch.where(dvds[..., 4] > 0, psi1_min, psi1_max)
            a2 = torch.where(dvds[..., 8] > 0, a2_min, a2_max)
            psi2 = torch.where(dvds[..., 9] > 0, psi2_min, psi2_max)

        return torch.cat((a1[..., None], psi1[..., None], a2[..., None], psi2[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [
                -6.0, -1.4, 0.0, 6.5, 0.0, 
                -6.0, 1.4, -math.pi, 0.0, 0.0
            ],
            'state_labels': [
                r'$x_1$', r'$y_1$', r'$\theta_1$', r'$v_1$', r'$\phi_1$',
                r'$x_2$', r'$y_2$', r'$\theta_2$', r'$v_2$', r'$\phi_2$',
            ],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

class ReachAvoidRocketLanding(Dynamics):
    def __init__(self):
        super().__init__(
            name = "ReachAvoidRocketLanding",loss_type='brat_hjivi', set_mode='reach',
            state_dim=6, input_dim=7, control_dim=2, disturbance_dim=0,
            state_mean=[0.0, 80.0, 0.0, 0.0, 0.0, 0.0],
            state_var=[150.0, 70.0, 1.2*math.pi, 200.0, 200.0, 10.0],
            value_mean=0.0,
            value_var=1.0,
            value_normto=0.02,
            deepreach_model="exact",
        )

    def state_test_range(self):
        return [
            [-150, 150],
            [10, 150],
            [-math.pi, math.pi],
            [-200, 200],
            [-200, 200],
            [-10, 10],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state 

    # \dot x = v_x
    # \dot y = v_y
    # \dot th = w
    # \dot v_x = u1 * cos(th) - u2 sin(th)
    # \dot v_y = u1 * sin(th) + u2 cos(th) - 9.81
    # \dot w = 0.3 * u1
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 3]
        dsdt[..., 1] = state[..., 4]
        dsdt[..., 2] = state[..., 5]
        dsdt[..., 3] = control[..., 0]*torch.cos(state[..., 2]) - control[..., 1]*torch.sin(state[..., 2])
        dsdt[..., 4] = control[..., 0]*torch.sin(state[..., 2]) + control[..., 1]*torch.cos(state[..., 2]) - 9.81
        dsdt[..., 5] = 0.3*control[..., 0]
        return dsdt

    def reach_fn(self, state):
        # Only target set in the xy direction
        # Target set position in x direction
        dist_x = torch.abs(state[..., 0]) - 20.0 #[-20, 150] boundary_fn range

        # Target set position in y direction
        dist_y = state[..., 1] - 20.0  #[-10, 130] boundary_fn range

        # First compute the target function as you normally would but then normalize it later.
        max_dist = torch.max(dist_x, dist_y)
        return torch.where((max_dist >= 0), max_dist/150.0, max_dist/10.0)

    def avoid_fn(self, state):
        # distance to floor
        dist_y = state[..., 1]

        # distance to wall
        wall_left = -30
        wall_right = -20
        wall_bottom = 0
        wall_top = 100
        dist_left = wall_left - state[..., 0]
        dist_right = state[..., 0] - wall_right
        dist_bottom = wall_bottom - state[..., 1]
        dist_top = state[..., 1] - wall_top
        dist_wall_x = torch.max(dist_left, dist_right)
        dist_wall_y = torch.max(dist_bottom, dist_top)
        dist_wall = torch.norm(torch.cat((torch.max(torch.tensor(0), dist_wall_x).unsqueeze(-1), torch.max(torch.tensor(0), dist_wall_y).unsqueeze(-1)), dim=-1), dim=-1) + torch.min(torch.tensor(0), torch.max(dist_wall_x, dist_wall_y))

        return torch.min(dist_y, dist_wall)

    def boundary_fn(self, state):
        return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))

    def sample_target_state(self, num_samples):
        target_state_range = self.state_test_range()
        target_state_range[0] = [-20, 20] # y in [-20, 20]
        target_state_range[1] = [10, 20]  # z in [10, 20]
        target_state_range = torch.tensor(target_state_range)
        return target_state_range[:, 0] + torch.rand(num_samples, self.state_dim)*(target_state_range[:, 1] - target_state_range[:, 0])

    def cost_fn(self, state_traj):
        # return min_t max{l(x(t)), max_k_up_to_t{-g(x(k))}}, where l(x) is reach_fn, g(x) is avoid_fn 
        reach_values = self.reach_fn(state_traj)
        avoid_values = self.avoid_fn(state_traj)
        return torch.min(torch.maximum(reach_values, torch.cummax(-avoid_values, dim=-1).values), dim=-1).values

    def hamiltonian(self, state, dvds):
        # Control Hamiltonian
        u1_coeff = dvds[..., 3] * torch.cos(state[..., 2]) + dvds[..., 4] * torch.sin(state[..., 2]) + 0.3 * dvds[..., 5]
        u2_coeff = -dvds[..., 3] * torch.sin(state[..., 2]) + dvds[..., 4] * torch.cos(state[..., 2])
        ham_ctrl = -250.0 * torch.sqrt(u1_coeff * u1_coeff + u2_coeff * u2_coeff)
        # Constant Hamiltonian
        ham_constant = dvds[..., 0] * state[..., 3] + dvds[..., 1] * state[..., 4] + \
                      dvds[..., 2] * state[..., 5]  - dvds[..., 4] * 9.81
        # Compute the Hamiltonian
        ham_vehicle = ham_ctrl + ham_constant
        return ham_vehicle

    def optimal_control(self, state, dvds):
        u1_coeff = dvds[..., 3] * torch.cos(state[..., 2]) + dvds[..., 4] * torch.sin(state[..., 2]) + 0.3 * dvds[..., 5]
        u2_coeff = -dvds[..., 3] * torch.sin(state[..., 2]) + dvds[..., 4] * torch.cos(state[..., 2])
        opt_angle = torch.atan2(u2_coeff, u1_coeff) + math.pi
        return torch.cat((250.0 * torch.cos(opt_angle)[..., None], 250.0 * torch.sin(opt_angle)[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [-100, 120, 0, 150, -5, 0.0],
            'state_labels': ['x', 'y', r'$\theta$', r'$v_x$', r'$v_y$', r'$\omega'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 4,
        }

class RocketLanding(Dynamics):
    def __init__(self):
        super().__init__(
            name="RocketLanding",loss_type='brt_hjivi', set_mode='reach',
            state_dim=6, input_dim=8, control_dim=2, disturbance_dim=0,
            state_mean=[0.0, 80.0, 0.0, 0.0, 0.0, 0.0],
            state_var=[150.0, 70.0, 1.2*math.pi, 200.0, 200.0, 10.0],
            value_mean=0.0,
            value_var=1.0,
            value_normto=0.02,
            deepreach_model="exact",
        )

    # convert model input to real coord
    def input_to_coord(self, input):
        input = input[..., :-1]
        coord = input.clone()
        coord[..., 1:] = (input[..., 1:] * self.state_var.to(device=input.device)) + self.state_mean.to(device=input.device)
        return coord

    # convert real coord to model input
    def coord_to_input(self, coord):
        input = coord.clone()
        input[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)) / self.state_var.to(device=coord.device)
        input = torch.cat((input, torch.zeros((*input.shape[:-1], 1), device=input.device)), dim=-1)
        return input

    # convert model io to real value
    def io_to_value(self, input, output):
        if self.deepreach_model=="diff":
            return (output * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        else:
            return (output * self.value_var / self.value_normto) + self.value_mean

    # convert model io to real dv
    def io_to_dv(self, input, output):
        dodi = diff_operators.jacobian(output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)[..., :-1]

        if self.deepreach_model=="diff":
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]

            dvds_term1 = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        
        else:
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
            dvds = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
        
        return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)


    def state_test_range(self):
        return [
            [-150, 150],
            [10, 150],
            [-math.pi, math.pi],
            [-200, 200],
            [-200, 200],
            [-10, 10],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state 

    # \dot x = v_x
    # \dot y = v_y
    # \dot th = w
    # \dot v_x = u1 * cos(th) - u2 sin(th)
    # \dot v_y = u1 * sin(th) + u2 cos(th) - 9.81
    # \dot w = 0.3 * u1
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 3]
        dsdt[..., 1] = state[..., 4]
        dsdt[..., 2] = state[..., 5]
        dsdt[..., 3] = control[..., 0]*torch.cos(state[..., 2]) - control[..., 1]*torch.sin(state[..., 2])
        dsdt[..., 4] = control[..., 0]*torch.sin(state[..., 2]) + control[..., 1]*torch.cos(state[..., 2]) - 9.81
        dsdt[..., 5] = 0.3*control[..., 0]
        return dsdt

    def boundary_fn(self, state):
        # Only target set in the yz direction
        # Target set position in y direction
        dist_y = torch.abs(state[..., 0]) - 20.0 #[-20, 150] boundary_fn range

        # Target set position in z direction
        dist_z = state[..., 1] - 20.0  #[-10, 130] boundary_fn range

        # First compute the l(x) as you normally would but then normalize it later.
        lx = torch.max(dist_y, dist_z)
        return torch.where((lx >= 0), lx/150.0, lx/10.0)

    def sample_target_state(self, num_samples):
        target_state_range = self.state_test_range()
        target_state_range[0] = [-20, 20] # y in [-20, 20]
        target_state_range[1] = [10, 20]  # z in [10, 20]
        target_state_range = torch.tensor(target_state_range)
        return target_state_range[:, 0] + torch.rand(num_samples, self.state_dim)*(target_state_range[:, 1] - target_state_range[:, 0])

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        # Control Hamiltonian
        u1_coeff = dvds[..., 3] * torch.cos(state[..., 2]) + dvds[..., 4] * torch.sin(state[..., 2]) + 0.3 * dvds[..., 5]
        u2_coeff = -dvds[..., 3] * torch.sin(state[..., 2]) + dvds[..., 4] * torch.cos(state[..., 2])
        ham_ctrl = -250.0 * torch.sqrt(u1_coeff * u1_coeff + u2_coeff * u2_coeff)
        # Constant Hamiltonian
        ham_constant = dvds[..., 0] * state[..., 3] + dvds[..., 1] * state[..., 4] + \
                      dvds[..., 2] * state[..., 5]  - dvds[..., 4] * 9.81
        # Compute the Hamiltonian
        ham_vehicle = ham_ctrl + ham_constant
        return ham_vehicle
    
    def optimal_control(self, state, dvds):
        u1_coeff = dvds[..., 3] * torch.cos(state[..., 2]) + dvds[..., 4] * torch.sin(state[..., 2]) + 0.3 * dvds[..., 5]
        u2_coeff = -dvds[..., 3] * torch.sin(state[..., 2]) + dvds[..., 4] * torch.cos(state[..., 2])
        opt_angle = torch.atan2(u2_coeff, u1_coeff) + math.pi
        return torch.cat((250.0 * torch.cos(opt_angle)[..., None], 250.0 * torch.sin(opt_angle)[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [-100, 120, 0, 150, -5, 0.0],
            'state_labels': ['x', 'y', r'$\theta$', r'$v_x$', r'$v_y$', r'$\omega'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 4,
        }

# class Quadrotor(Dynamics):
#     def __init__(self, collisionR:float, collective_thrust_max:float, set_mode:str):
#         self.collective_thrust_max = collective_thrust_max
#         self.m=1 #mass
#         self.arm_l=0.17
#         self.CT=1
#         self.CM=0.016
#         self.Gz=-9.8

#         self.collective_thrust_max = collective_thrust_max
#         self.collisionR = collisionR


#         super().__init__(
#             name = "Quadrotor", loss_type='brt_hjivi', set_mode=set_mode,
#             state_dim=13, input_dim=14, control_dim=4, disturbance_dim=0,
#             state_mean=[0 for i in range(13)], 
#             state_var=[1.5, 1.5, 1.5, 1, 1, 1, 1, 10, 10 ,10 ,10 ,10 ,10],
#             value_mean=(math.sqrt(1.5**2+1.5**2+1.5**2)-2*self.collisionR)/2, 
#             value_var=math.sqrt(1.5**2+1.5**2+1.5**2), 
#             value_normto=0.02,
#             deepreach_model="exact"
#         )

#     def state_test_range(self):
#         return [
#             [-1.5, 1.5],
#             [-1.5, 1.5],
#             [-1.5, 1.5],
#             [-1, 1],
#             [-1, 1],
#             [-1, 1],
#             [-1, 1],
#             [-10, 10],
#             [-10, 10],
#             [-10, 10],
#             [-10, 10],
#             [-10, 10],
#             [-10, 10],
#         ]

#     def equivalent_wrapped_state(self, state):
#         wrapped_state = torch.clone(state)
#         return wrapped_state

#     # Dubins3D dynamics
#     # \dot x    = v \cos \theta
#     # \dot y    = v \sin \theta
#     # \dot \theta = u
#     def dsdt(self, state, control, disturbance):
#         qw = state[..., 3] * 1.0
#         qx = state[..., 4] * 1.0
#         qy = state[..., 5] * 1.0
#         qz = state[..., 6] * 1.0
#         vx = state[..., 7] * 1.0
#         vy = state[..., 8] * 1.0
#         vz = state[..., 9] * 1.0
#         wx = state[..., 10] * 1.0
#         wy = state[..., 11] * 1.0
#         wz = state[..., 12] * 1.0
#         u1 = control[...,0] * 1.0
#         u2 = control[...,1] * 1.0
#         u3 = control[...,2] * 1.0
#         u4 = control[...,3] * 1.0


#         dsdt = torch.zeros_like(state)
#         dsdt[..., 0] = vx
#         dsdt[..., 1] = vy
#         dsdt[..., 2] = vz
#         dsdt[..., 3] = -(wx*qx+wy*qy+wz*qz)/2.0 
#         dsdt[..., 4] =  (wx*qw+wz*qy-wy*qz)/2.0
#         dsdt[..., 5] = (wy*qw-wz*qx+wx*qz)/2.0
#         dsdt[..., 6] = (wz*qw+wy*qx-wx*qy)/2.0
#         dsdt[..., 7] = 2*(qw*qy+qx*qz)*self.CT/self.m*(u1+u2+u3+u4)
#         dsdt[..., 8] =2*(-qw*qx+qy*qz)*self.CT/self.m*(u1+u2+u3+u4)
#         dsdt[..., 9] =self.Gz+(1-2*torch.pow(qx,2)-2*torch.pow(qy,2))*self.CT/self.m*(u1+u2+u3+u4)
#         dsdt[..., 10] = 4*math.sqrt(2)*self.CT*(u1-u2-u3+u4)/(3*self.arm_l*self.m)-5*wy*wz/9.0
#         dsdt[..., 11] = 4*math.sqrt(2)*self.CT*(-u1-u2+u3+u4)/(3*self.arm_l*self.m)+5*wx*wz/9.0
#         dsdt[..., 12] =12*self.CT*self.CM/(7*self.arm_l**2*self.m)*(u1-u2+u3-u4)
#         return dsdt

#     def boundary_fn(self, state):
#         return torch.norm(state[..., :3], dim=-1) - self.collisionR

#     def sample_target_state(self, num_samples):
#         raise NotImplementedError

#     def cost_fn(self, state_traj):
#         return torch.min(self.boundary_fn(state_traj), dim=-1).values

#     def hamiltonian(self, state, dvds):
#         if self.set_mode == 'reach':
#             raise NotImplementedError

#         elif self.set_mode == 'avoid':
#             qw = state[..., 3] * 1.0
#             qx = state[..., 4] * 1.0
#             qy = state[..., 5] * 1.0
#             qz = state[..., 6] * 1.0
#             vx = state[..., 7] * 1.0
#             vy = state[..., 8] * 1.0
#             vz = state[..., 9] * 1.0
#             wx = state[..., 10] * 1.0
#             wy = state[..., 11] * 1.0
#             wz = state[..., 12] * 1.0


#             C1=2*(qw*qy+qx*qz)*self.CT/self.m
#             C2=2*(-qw*qx+qy*qz)*self.CT/self.m
#             C3=(1-2*torch.pow(qx,2)-2*torch.pow(qy,2))*self.CT/self.m
#             C4=4*math.sqrt(2)*self.CT/(3*self.arm_l*self.m)
#             C5=4*math.sqrt(2)*self.CT/(3*self.arm_l*self.m)
#             C6=12*self.CT*self.CM/(7*self.arm_l**2*self.m)

#             # Compute the hamiltonian for the quadrotor
#             ham= dvds[..., 0]*vx + dvds[..., 1]*vy+ dvds[..., 2]*vz
#             ham+= -dvds[..., 3]* (wx*qx+wy*qy+wz*qz)/2.0 
#             ham+= dvds[..., 4]*(wx*qw+wz*qy-wy*qz)/2.0
#             ham+= dvds[..., 5]*(wy*qw-wz*qx+wx*qz)/2.0
#             ham+= dvds[..., 6]*(wz*qw+wy*qx-wx*qy)/2.0
#             ham+= dvds[..., 9]*-9.8
#             ham+= -dvds[..., 10]*5*wy*wz/9.0+ dvds[..., 11]*5*wx*wz/9.0

#             ham+=torch.abs(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
#                 +dvds[..., 10]*C4-dvds[..., 11]*C5+dvds[..., 12]*C6)*self.collective_thrust_max

#             ham+=torch.abs(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
#                 -dvds[..., 10]*C4-dvds[..., 11]*C5-dvds[..., 12]*C6)*self.collective_thrust_max

#             ham+=torch.abs(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
#                 -dvds[..., 10]*C4+dvds[..., 11]*C5+dvds[..., 12]*C6)*self.collective_thrust_max

#             ham+=torch.abs(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
#                 +dvds[..., 10]*C4+dvds[..., 11]*C5-dvds[..., 12]*C6)*self.collective_thrust_max

#             return ham

#     def optimal_control(self, state, dvds):
#         if self.set_mode == 'reach':
#             raise NotImplementedError
#         elif self.set_mode == 'avoid':
#             qw = state[..., 3] * 1.0
#             qx = state[..., 4] * 1.0
#             qy = state[..., 5] * 1.0
#             qz = state[..., 6] * 1.0


#             C1=2*(qw*qy+qx*qz)*self.CT/self.m
#             C2=2*(-qw*qx+qy*qz)*self.CT/self.m
#             C3=(1-2*torch.pow(qx,2)-2*torch.pow(qy,2))*self.CT/self.m
#             C4=4*math.sqrt(2)*self.CT/(3*self.arm_l*self.m)
#             C5=4*math.sqrt(2)*self.CT/(3*self.arm_l*self.m)
#             C6=12*self.CT*self.CM/(7*self.arm_l**2*self.m)


#             u1=self.collective_thrust_max*torch.sign(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
#                 +dvds[..., 10]*C4-dvds[..., 11]*C5+dvds[..., 12]*C6)
#             u2=self.collective_thrust_max*torch.sign(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
#                 -dvds[..., 10]*C4-dvds[..., 11]*C5-dvds[..., 12]*C6)
#             u3=self.collective_thrust_max*torch.sign(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
#                 -dvds[..., 10]*C4+dvds[..., 11]*C5+dvds[..., 12]*C6)
#             u4=self.collective_thrust_max*torch.sign(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
#                 +dvds[..., 10]*C4+dvds[..., 11]*C5-dvds[..., 12]*C6)

#         return torch.cat((u1[..., None], u2[..., None], u3[..., None], u4[..., None]), dim=-1)

#     def optimal_disturbance(self, state, dvds):
#         return 0

#     def plot_config(self):
#         return {
#             'state_slices': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             'state_labels': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
#             'x_axis_idx': 0,
#             'y_axis_idx': 2,
#             'z_axis_idx': 7,
#         }


class Quadrotor10D(Dynamics):
    def __init__(self, collisionR: float, set_mode: str, N: int):  # simpler quadrotor

        # self.d0=10.0
        # self.d1=8.0
        # self.n0=10.0
        self.d0=7.0
        self.d1=4.0
        self.n0=12.0
        # self.g=6.0
        self.g=0.91
        self.u_max=math.pi/4
        self.u3_max=1.0

        self.collisionR = collisionR
        self.N = N

        super().__init__(
            name='Quadrotor10D', loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=10, input_dim=11, control_dim=3, disturbance_dim=0,
            state_mean=[0 for i in range(10)],
            # state_var=[3.0, 3.0, 1.50, 3.0, 3.0, 3.0, 1.50, 3.0, 1.0, 3.0],
            # value_mean=(math.sqrt(3.0**2 + 3.0**2) -
            #             2 * self.collisionR) / 2,
            # value_var=math.sqrt(3.0**2 + 3.0**2),
            state_var=[4.0, 3.0, 1.50, 6.0, 4.0, 3.0, 1.50, 6.0, 2.0, 2.0],
            value_mean=(math.sqrt(4.0**2 + 4.0**2) -
                        2 * self.collisionR) / 2,
            value_var=math.sqrt(4.0**2 + 4.0**2),
            value_normto=0.02,
            deepreach_model='exact'
        )
    
    
    def control_range(self, state):
        return [[-self.u_max, self.u_max],
                [-self.u_max, self.u_max],
                [-self.u3_max, self.u3_max]]

    def state_test_range(self):
        return [
            [-4.0, 4.0],
            [-3.0, 3.0],
            [-1.50, 1.50],
            [-6.0, 6.0],
            [-4.0, 4.0],
            [-3.0, 3.0],
            [-1.50, 1.50],
            [-6.0, 6.0],
            [-2.0, 2.0],
            [-2.0, 2.0],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # return wrapped_state
        return wrapped_state
    
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 1]*1.0
        dsdt[..., 1] = torch.tan(state[..., 2])*self.g
        dsdt[..., 2] = -self.d1*state[..., 2] + state[..., 3]*1.0
        dsdt[..., 3] = -self.d0*state[..., 2] + self.n0 * control[...,0]

        dsdt[..., 4] = state[..., 5]*1.0
        dsdt[..., 5] = torch.tan(state[..., 6])*self.g
        dsdt[..., 6] = -self.d1*state[..., 6] + state[..., 7]*1.0
        dsdt[..., 7] = -self.d0*state[..., 6] + self.n0 * control[...,1]

        dsdt[..., 8] = state[..., 9]*1.0
        dsdt[..., 9] = control[...,2]*1.0
        return dsdt

    def boundary_fn(self, state):
        '''for cylinder with point-mass collision'''
        dist = torch.norm(state[..., [0,4]], dim=-1)
        return torch.maximum(dist, torch.zeros_like(dist)) - self.collisionR


    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError

        elif self.set_mode == 'avoid':
            ham = dvds[..., 0] * state[..., 1] + dvds[..., 1] * torch.tan(state[..., 2])*self.g + dvds[..., 2] * (-self.d1*state[..., 2] + state[..., 3]*1.0) + \
                  dvds[..., 3] * (-self.d0*state[..., 2]) + dvds[..., 4] * state[..., 5] + dvds[..., 5] * torch.tan(state[..., 6])*self.g + \
                  dvds[..., 6] * (-self.d1*state[..., 6] + state[..., 7]*1.0) + dvds[..., 7] * (-self.d0*state[..., 6]) + dvds[..., 8] * state[..., 9]
            ham += torch.abs(dvds[..., 3]*self.n0)*self.u_max + torch.abs(dvds[..., 7]*self.n0)*self.u_max + torch.abs(dvds[..., 9])*self.u3_max
            return ham

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError
        elif self.set_mode == 'avoid':
            
            u1 = self.u_max * torch.sign(dvds[..., 3]*self.n0)
            u2 = self.u_max * torch.sign(dvds[..., 7]*self.n0)
            u3 = self.u3_max * torch.sign(dvds[..., 9])

        return torch.cat((u1[..., None], u2[..., None], u3[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return torch.zeros(1)


    def plot_config(self):
        
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        return {
            'state_slices': [0, 0, 1, 5, 0, 2, 1, 2, 0, 0.4],
            'state_labels': ['x', 'vx', 'pitch', 'wy', 'y', 'vy', 'roll', 'wx', 'z', 'vz'],
            'x_axis_idx': 0,
            'y_axis_idx': 4,
            'z_axis_idx': 1,
        }


class Quadrotor10DLambda(Dynamics):
    def __init__(self, collisionR: float, set_mode: str, N: int, dynamics_mode:str):  # simpler quadrotor
        self.d0=7.0
        self.d1=4.0
        self.n0=12.0

        self.g=0.91
        self.u_max=math.pi/4
        self.u3_max=1.0

        self.collisionR = collisionR
        self.N = N
        self.mode = dynamics_mode # choice: ["linear","lambda", "lambda_time"] 
        
        self.lin_pt=torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.hopf_model=None
        super().__init__(
            name='Quadrotor10DLambda', loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=11, input_dim=12, control_dim=3, disturbance_dim=0,
            state_mean=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5],
            # state_var=[3.0, 3.0, 1.50, 3.0, 3.0, 3.0, 1.50, 3.0, 1.0, 3.0],
            # value_mean=(math.sqrt(3.0**2 + 3.0**2) -
            #             2 * self.collisionR) / 2,
            # value_var=math.sqrt(3.0**2 + 3.0**2),
            state_var=[4.0, 3.0, 1.50, 6.0, 4.0, 3.0, 1.50, 6.0, 2.0, 2.0, 0.5],
            value_mean=(math.sqrt(4.0**2 + 4.0**2) -
                        2 * self.collisionR) / 2,
            value_var=math.sqrt(4.0**2 + 4.0**2),
            value_normto=0.02,
            deepreach_model='exact'
        )
    
    
    def control_range(self, state):
        return [[-self.u_max, self.u_max],
                [-self.u_max, self.u_max],
                [-self.u3_max, self.u3_max]]

    def state_test_range(self):
        return [
            [-4.0, 4.0],
            [-3.0, 3.0],
            [-1.50, 1.50],
            [-6.0, 6.0],
            [-4.0, 4.0],
            [-3.0, 3.0],
            [-1.50, 1.50],
            [-6.0, 6.0],
            [-2.0, 2.0],
            [-2.0, 2.0],
            [0.0, 1.0],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # return wrapped_state
        return wrapped_state
    
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 1]*1.0
        dsdt[..., 1] = (state[..., -1]*torch.tan(state[..., 2]) + (torch.ones_like(state[..., -1])-state[..., -1])*state[..., 2]) * self.g
        dsdt[..., 2] = -self.d1*state[..., 2] + state[..., 3]*1.0
        dsdt[..., 3] = -self.d0*state[..., 2] + self.n0 * control[...,0]

        dsdt[..., 4] = state[..., 5]*1.0
        dsdt[..., 5] = (state[..., -1]*torch.tan(state[..., 6]) + (torch.ones_like(state[..., -1])-state[..., -1])*state[..., 6])*self.g
        dsdt[..., 6] = -self.d1*state[..., 6] + state[..., 7]*1.0
        dsdt[..., 7] = -self.d0*state[..., 6] + self.n0 * control[...,1]

        dsdt[..., 8] = state[..., 9]*1.0
        dsdt[..., 9] = control[...,2]*1.0

        return dsdt

    def boundary_fn(self, state):
        '''for cylinder with point-mass collision'''
        dist = torch.norm(state[..., [0,4]], dim=-1)
        return torch.maximum(dist, torch.zeros_like(dist)) - self.collisionR


    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError

        elif self.set_mode == 'avoid':
            # ham = dvds[..., 0] * state[..., 1] + \
            #       dvds[..., 1] * (state[..., -1]*torch.tan(state[..., 2]) + (torch.ones_like(state[..., -1])-state[..., -1])*state[..., 2]) * self.g +\
            #       dvds[..., 2] * (-self.d1*state[..., 2] + state[..., 3]*1.0) + \
            #       dvds[..., 3] * (-self.d0*state[..., 2]) + dvds[..., 4] * state[..., 5] + \
            #       dvds[..., 5] * (state[..., -1]*torch.tan(state[..., 6]) + (torch.ones_like(state[..., -1])-state[..., -1])*state[..., 6])*self.g + \
            #       dvds[..., 6] * (-self.d1*state[..., 6] + state[..., 7]*1.0) + dvds[..., 7] * (-self.d0*state[..., 6]) + dvds[..., 8] * state[..., 9]
            
            
            # ham += torch.abs(dvds[..., 3]*self.n0)*self.u_max + torch.abs(dvds[..., 7]*self.n0)*self.u_max + torch.abs(dvds[..., 9])*self.u3_max


            ham = dvds[..., 0] * state[..., 1] + \
                  dvds[..., 2] * (-self.d1*state[..., 2] + state[..., 3]*1.0) + dvds[..., 3] * (-self.d0*state[..., 2]) + dvds[..., 4] * state[..., 5] + \
                  dvds[..., 6] * (-self.d1*state[..., 6] + state[..., 7]*1.0) + dvds[..., 7] * (-self.d0*state[..., 6]) + dvds[..., 8] * state[..., 9]
            
            ham += (dvds[..., 1] * (torch.tan(self.lin_pt[2])*self.g + 1/ torch.cos(self.lin_pt[2])*self.g * (state[..., 2]-self.lin_pt[2]))  + \
                   dvds[..., 5] * (torch.tan(self.lin_pt[6])*self.g + 1/ torch.cos(self.lin_pt[6])*self.g * (state[..., 6]-self.lin_pt[6])) ) * \
                   (torch.ones_like(state[...,-1])-state[...,-1])
            
            ham += ( dvds[..., 1] * torch.tan(state[..., 2])*self.g + dvds[..., 5] * torch.tan(state[..., 6])*self.g ) * state[...,-1]
            
            ham += torch.abs(dvds[..., 3]*self.n0)*self.u_max + torch.abs(dvds[..., 7]*self.n0)*self.u_max + torch.abs(dvds[..., 9])*self.u3_max
            return ham

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError
        elif self.set_mode == 'avoid':
            
            u1 = self.u_max * torch.sign(dvds[..., 3]*self.n0)
            u2 = self.u_max * torch.sign(dvds[..., 7]*self.n0)
            u3 = self.u3_max * torch.sign(dvds[..., 9])

        return torch.cat((u1[..., None], u2[..., None], u3[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return torch.zeros(1)

    # convert model io to real value
    def io_to_value(self, input, output):
        if self.deepreach_model=="diff":
            return (output * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepreach_model=="exact":
            return (output * input[..., 0] * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepreach_model=="exact_lambda":
            hopf_input = input*1.0
            hopf_input[...,-1] = -1.0 # lambda is always 0 for hopf input 
            hopf_model_results = self.hopf_model({'coords': hopf_input})
            self.hopf_out = hopf_model_results['model_out'].squeeze(dim=-1)
            self.hopf_in = hopf_model_results['model_in']
            self.hopf_values = (self.hopf_out * self.hopf_in[..., 0] * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(self.hopf_in)[..., 1:])
            return self.hopf_values + input[..., 0] * self.input_to_coord(input)[..., -1] * output * self.value_var / self.value_normto
           
        else:
            return (output * self.value_var / self.value_normto) + self.value_mean

    # convert model io to real dv
    def io_to_dv(self, input, output):
        dodi = diff_operators.jacobian(output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)

        if self.deepreach_model=="diff":
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]

            dvds_term1 = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        elif self.deepreach_model=="exact":
            dvdt = (self.value_var / self.value_normto) * \
                (input[..., 0]*dodi[..., 0] + output)

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:] * input[..., 0].unsqueeze(-1)
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        elif self.deepreach_model=="exact_lambda":
            state = self.input_to_coord(input)[..., 1:]
            hopf_jacobian=diff_operators.jacobian(self.hopf_values.unsqueeze(dim=-1), self.hopf_in)[0].squeeze(dim=-2)[...,:-1]

            dvdt =  hopf_jacobian[..., 0] + \
                (self.value_var / self.value_normto) * (input[..., 0]*dodi[..., 0] + output) * state[..., -1]
            
            dvds = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:] * input[..., 0].unsqueeze(-1) * state[..., -1].unsqueeze(-1)
            
            dvds[...,-1] = dvds[...,-1] + (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)[-1]) * output[..., -1] * input[..., 0]
            
            dvds[...,:-1] = dvds[...,:-1] +  hopf_jacobian[..., 1:] / self.state_var.to(device=dodi.device)[:-1]

        else:
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
            dvds = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
        
        return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

    def plot_config(self):
        if self.mode == "linear":
            return {
                'state_slices': [0, 0, 1, 5, 0, 2, 1, 2, 0, 0.4, 0.0],
                'state_labels': ['x', 'vx', 'pitch', 'wy', 'y', 'vy', 'roll', 'wx', 'z', 'vz', 'lambda'],
                'x_axis_idx': 0,
                'y_axis_idx': 4,
                'z_axis_idx': 1,
                }
        elif self.mode in ["lambda", "lambda_time"]:
            # return {
            #     'state_slices': [0, 3.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #     'state_labels': ['x', 'vx', 'pitch', 'wy', 'y', 'vy', 'roll', 'wx', 'z', 'vz', 'lambda'],
            #     'x_axis_idx': 0,
            #     'y_axis_idx': 4,
            #     'z_axis_idx': 10,
            # }
            return {
                'state_slices': [0, 0, 1, 5, 0, 2, 1, 2, 0, 0.4, 1.0], 
                'state_labels': ['x', 'vx', 'pitch', 'wy', 'y', 'vy', 'roll', 'wx', 'z', 'vz', 'lambda'],
                'x_axis_idx': 0,
                'y_axis_idx': 4,
                'z_axis_idx': 1,
                }
        else:
            raise NotImplementedError
    
class Quadrotor10DLinear(Dynamics):
    def __init__(self, collisionR: float, set_mode: str, N: int):  # simpler quadrotor

        # self.d0=10.0
        # self.d1=8.0
        # self.n0=10.0
        self.d0=7.0
        self.d1=4.0
        self.n0=12.0
        self.g=0.91
        self.u_max=math.pi/4
        self.u3_max=1.0

        self.collisionR = collisionR
        self.N = N
        self.lin_pt=torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.lin_ctrl_pt = torch.tensor([0., 0., 0.])

        super().__init__(
            name='Quadrotor10D', loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=10, input_dim=11, control_dim=3, disturbance_dim=0,
            state_mean=[0 for i in range(10)],
            state_var=[4.0, 3.0, 1.50, 6.0, 4.0, 3.0, 1.50, 6.0, 2.0, 2.0],
            value_mean=(math.sqrt(4.0**2 + 4.0**2) -
                        2 * self.collisionR) / 2,
            value_var=math.sqrt(4.0**2 + 4.0**2),
            value_normto=0.02,
            deepreach_model='exact'
        )
    
    
    def control_range(self, state):
        return [[-self.u_max, self.u_max],
                [-self.u_max, self.u_max],
                [-self.u3_max, self.u3_max]]

    def state_test_range(self):
        return [
            [-4.0, 4.0],
            [-3.0, 3.0],
            [-1.50, 1.50],
            [-6.0, 6.0],
            [-4.0, 4.0],
            [-3.0, 3.0],
            [-1.50, 1.50],
            [-6.0, 6.0],
            [-2.0, 2.0],
            [-2.0, 2.0],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # return wrapped_state
        return wrapped_state
    
    # def dsdt(self, state, control, disturbance):
    #     dsdt = torch.zeros_like(state)
    #     dsdt[..., 0] = state[..., 1]*1.0
    #     dsdt[..., 1] = torch.tan(self.lin_pt[2])*self.g + 1/ torch.cos(self.lin_pt[2])*self.g * (state[..., 2]-self.lin_pt[2])
    #     dsdt[..., 2] = -self.d1*state[..., 2] + state[..., 3]*1.0
    #     dsdt[..., 3] = -self.d0*state[..., 2] + self.n0 * control[...,0]

    #     dsdt[..., 4] = state[..., 5]*1.0
    #     dsdt[..., 5] = torch.tan(self.lin_pt[6])*self.g + 1/ torch.cos(self.lin_pt[6])*self.g * (state[..., 6]-self.lin_pt[6])
    #     dsdt[..., 6] = -self.d1*state[..., 6] + state[..., 7]*1.0
    #     dsdt[..., 7] = -self.d0*state[..., 6] + self.n0 * control[...,1]

    #     dsdt[..., 8] = state[..., 9]*1.0
    #     dsdt[..., 9] = control[...,2]*1.0
    #     return dsdt
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 1]*1.0
        dsdt[..., 1] = torch.tan(state[..., 2])*self.g
        dsdt[..., 2] = -self.d1*state[..., 2] + state[..., 3]*1.0
        dsdt[..., 3] = -self.d0*state[..., 2] + self.n0 * control[...,0]

        dsdt[..., 4] = state[..., 5]*1.0
        dsdt[..., 5] = torch.tan(state[..., 6])*self.g
        dsdt[..., 6] = -self.d1*state[..., 6] + state[..., 7]*1.0
        dsdt[..., 7] = -self.d0*state[..., 6] + self.n0 * control[...,1]

        dsdt[..., 8] = state[..., 9]*1.0
        dsdt[..., 9] = control[...,2]*1.0
        return dsdt

    def boundary_fn(self, state):
        '''for cylinder with point-mass collision'''
        dist = torch.norm(state[..., [0,4]], dim=-1)
        return torch.maximum(dist, torch.zeros_like(dist)) - self.collisionR

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError

        elif self.set_mode == 'avoid':
            ham = dvds[..., 0] * state[..., 1] + dvds[..., 1] * (torch.tan(self.lin_pt[2])*self.g + 1/ torch.cos(self.lin_pt[2])*self.g * (state[..., 2]-self.lin_pt[2]))+ \
                  dvds[..., 2] * (-self.d1*state[..., 2] + state[..., 3]*1.0) + dvds[..., 3] * (-self.d0*state[..., 2]) + dvds[..., 4] * state[..., 5] + \
                  dvds[..., 5] * (torch.tan(self.lin_pt[6])*self.g + 1/ torch.cos(self.lin_pt[6])*self.g * (state[..., 6]-self.lin_pt[6]))  + \
                  dvds[..., 6] * (-self.d1*state[..., 6] + state[..., 7]*1.0) + dvds[..., 7] * (-self.d0*state[..., 6]) + dvds[..., 8] * state[..., 9]
            
            ham += torch.abs(dvds[..., 3]*self.n0)*self.u_max + torch.abs(dvds[..., 7]*self.n0)*self.u_max + torch.abs(dvds[..., 9])*self.u3_max
            return ham

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError
        elif self.set_mode == 'avoid':
            
            u1 = self.u_max * torch.sign(dvds[..., 3]*self.n0)
            u2 = self.u_max * torch.sign(dvds[..., 7]*self.n0)
            u3 = self.u3_max * torch.sign(dvds[..., 9])

        return torch.cat((u1[..., None], u2[..., None], u3[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return torch.zeros(1)


    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'state_labels': ['x', 'vx', 'pitch', 'wy', 'y', 'vy', 'roll', 'wx', 'z', 'vz'],
            'x_axis_idx': 0,
            'y_axis_idx': 4,
            'z_axis_idx': 1,
        }

class Quadrotor(Dynamics):
    def __init__(self, collisionR: float, collective_thrust_max: float,  set_mode: str, N: int):  # simpler quadrotor
        self.collective_thrust_max = collective_thrust_max
        # self.body_rate_acc_max = body_rate_acc_max
        self.m = 1  # mass
        self.arm_l = 0.17
        self.CT = 1
        self.CM = 0.016
        self.Gz = -9.8

        self.dwx_max = 8
        self.dwy_max = 8
        self.dwz_max = 4
        self.dist_dwx_max = 0
        self.dist_dwy_max = 0
        self.dist_dwz_max = 0
        self.dist_f = 0

        self.collisionR = collisionR
        self.N = N

        super().__init__(
            name='Quadrotor', loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=13, input_dim=14, control_dim=4, disturbance_dim=0,
            state_mean=[0 for i in range(13)],
            state_var=[3.0, 3.0, 3.0, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5],
            value_mean=(math.sqrt(3.0**2 + 3.0**2) -
                        2 * self.collisionR) / 2,
            value_var=math.sqrt(3.0**2 + 3.0**2),
            value_normto=0.02,
            deepreach_model='exact'
        )
    def normalize_q(self, x):
        # normalize quaternion
        normalized_x = x*1.0
        q_tensor = x[..., 3:7]
        q_tensor = torch.nn.functional.normalize(
            q_tensor, p=2,dim=-1)  # normalize quaternion
        normalized_x[..., 3:7] = q_tensor
        return normalized_x
    
    def control_range(self, state):
        return [[-self.collective_thrust_max, self.collective_thrust_max],
                [-self.dwx_max, self.dwx_max],
                [-self.dwy_max, self.dwy_max],
                [-self.dwz_max, self.dwz_max]]

    def state_test_range(self):
        return [
            [-3.0, 3.0],
            [-3.0, 3.0],
            [-3.0, 3.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # return wrapped_state
        return self.normalize_q(wrapped_state)

    def dsdt(self, state, control, disturbance):
        qw = state[..., 3] * 1.0
        qx = state[..., 4] * 1.0
        qy = state[..., 5] * 1.0
        qz = state[..., 6] * 1.0
        vx = state[..., 7] * 1.0
        vy = state[..., 8] * 1.0
        vz = state[..., 9] * 1.0
        wx = state[..., 10] * 1.0
        wy = state[..., 11] * 1.0
        wz = state[..., 12] * 1.0
        f = (control[..., 0]) * 1.0

        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = vx
        dsdt[..., 1] = vy
        dsdt[..., 2] = vz
        dsdt[..., 3] = -(wx * qx + wy * qy + wz * qz) / 2.0
        dsdt[..., 4] = (wx * qw + wz * qy - wy * qz) / 2.0
        dsdt[..., 5] = (wy * qw - wz * qx + wx * qz) / 2.0
        dsdt[..., 6] = (wz * qw + wy * qx - wx * qy) / 2.0
        dsdt[..., 7] = 2 * (qw * qy + qx * qz) * self.CT / \
            self.m * f
        dsdt[..., 8] = 2 * (-qw * qx + qy * qz) * self.CT / \
            self.m * f
        dsdt[..., 9] = self.Gz + (1 - 2 * torch.pow(qx, 2) - 2 *
                                  torch.pow(qy, 2)) * self.CT / self.m * f
        dsdt[..., 10] = (control[..., 1]
                         ) * 1.0 - 5 * wy * wz / 9.0
        dsdt[..., 11] = (control[..., 2]
                         ) * 1.0 + 5 * wx * wz / 9.0
        dsdt[..., 12] = (control[..., 3]) * 1.0

        return dsdt

    def boundary_fn(self, state):
        '''for cylinder with point-mass collision'''
        dist = torch.norm(state[..., :2], dim=-1)
        return torch.maximum(dist, torch.zeros_like(dist)) - self.collisionR
        '''for cylinder with full body collision'''
        # create normal vector
        # v = torch.zeros_like(state[..., 4:7])
        # v[..., 2] = 1
        # v = quaternion.quaternion_apply(state[..., 3:7], v)
        # vx = v[..., 0]
        # vy = v[..., 1]
        # vz = v[..., 2]
        # # compute vector from center of quadrotor to the center of cylinder
        # px = state[..., 0]
        # py = state[..., 1]

        # # get full body distance
        # dist = torch.norm(state[..., :2], dim=-1)
        # # return dist- self.collisionR
        # dist = dist- torch.sqrt((self.arm_l**2*px**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2)
        #                    + (self.arm_l**2*py**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2))
        # return torch.maximum(dist, torch.zeros_like(dist)) - self.collisionR


    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError

        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0
            vx = state[..., 7] * 1.0
            vy = state[..., 8] * 1.0
            vz = state[..., 9] * 1.0
            wx = state[..., 10] * 1.0
            wy = state[..., 11] * 1.0
            wz = state[..., 12] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            # Compute the hamiltonian for the quadrotor
            ham = dvds[..., 0] * vx + dvds[..., 1] * vy + dvds[..., 2] * vz
            ham += -dvds[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
            ham += dvds[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
            ham += dvds[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
            ham += dvds[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0
            ham += dvds[..., 9] * self.Gz
            ham += -dvds[..., 10] * 5 * wy * wz / \
                9.0 + dvds[..., 11] * 5 * wx * wz / 9.0

            ham += torch.abs(dvds[..., 7] * c1 + dvds[..., 8] *
                             c2 + dvds[..., 9] * c3) * self.collective_thrust_max

            ham += torch.abs(dvds[..., 10]) * self.dwx_max + torch.abs(
                dvds[..., 11]) * self.dwy_max + torch.abs(dvds[..., 12]) * self.dwz_max

            # ham = ham+ torch.clamp(torch.abs(dvds[..., 7] * c1)* self.collective_thrust_max,-0.1,0.1) + \
            #         torch.clamp(torch.abs(dvds[..., 8] * c2)* self.collective_thrust_max,-0.1,0.1) + \
            #         torch.clamp(torch.abs(dvds[..., 9] * c3)* self.collective_thrust_max + dvds[..., 9] * self.Gz,-0.1,0.1)
            # ham = ham+ torch.clamp(-dvds[..., 10] * 5 * wy * wz /  9.0+ torch.abs(dvds[..., 10]) * (self.dwx_max),-0.1,0.1) + \
            #           torch.clamp(dvds[..., 11] * 5 * wx * wz / 9.0+ torch.abs( dvds[..., 11]) * (self.dwy_max) ,-0.1,0.1) + \
            #         torch.clamp(torch.abs(dvds[..., 12]) * (self.dwz_max),-0.1,0.1)

            return ham

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError
        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            u1 = self.collective_thrust_max * \
                torch.sign(dvds[..., 7] * c1 + dvds[..., 8] *
                           c2 + dvds[..., 9] * c3)
            u2 = self.dwx_max * torch.sign(dvds[..., 10])
            u3 = self.dwy_max * torch.sign(dvds[..., 11])
            u4 = self.dwz_max * torch.sign(dvds[..., 12])

        return torch.cat((u1[..., None], u2[..., None], u3[..., None], u4[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return torch.zeros(1)


    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'state_labels': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 7,
        }
    
class QuadrotorLinear2(Dynamics):
    def __init__(self, collisionR: float, collective_thrust_max: float,  set_mode: str, N: int):  # simpler quadrotor
        self.collective_thrust_max = collective_thrust_max
        self.m = 1  # mass
        self.arm_l = 0.17
        self.CT = 1
        self.CM = 0.016
        self.Gz = -9.8

        self.dwx_max = 8
        self.dwy_max = 8
        self.dwz_max = 4
        # self.dwx_max = 4
        # self.dwy_max = 4
        # self.dwz_max = 2
        self.dist_dwx_max = 0
        self.dist_dwy_max = 0
        self.dist_dwz_max = 0
        self.dist_f = 0

        self.collisionR = collisionR
        self.N = N
        
        # self.lin_ctrl_pt = torch.tensor([self.m * self.Gz / self.CT/3, 0., 0., 0.])
        self.lin_ctrl_pt = torch.tensor([self.collective_thrust_max, 0., 0., 0.])

        super().__init__(
            name='QuadrotorLinear', loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=13, input_dim=14, control_dim=4, disturbance_dim=0,
            state_mean=[0 for i in range(13)],
            state_var=[3.0, 3.0, 3.0, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5],
            value_mean=(math.sqrt(3.0**2 + 3.0**2) -
                        2 * self.collisionR) / 2,
            value_var=math.sqrt(3.0**2 + 3.0**2),
            value_normto=0.02,
            deepreach_model='exact'
        )
    def normalize_q(self, x):
        # normalize quaternion
        normalized_x = x*1.0
        q_tensor = x[..., 3:7]*1.0
        q_tensor = torch.nn.functional.normalize(
            q_tensor, p=2,dim=-1)  # normalize quaternion
        normalized_x[..., 3:7] = q_tensor
        return normalized_x
    
    def control_range(self, state):
        return [[-self.collective_thrust_max, self.collective_thrust_max],
                [-self.dwx_max, self.dwx_max],
                [-self.dwy_max, self.dwy_max],
                [-self.dwz_max, self.dwz_max]]

    def state_test_range(self):
        return [
            [-3.0, 3.0],
            [-3.0, 3.0],
            [-3.0, 3.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # return wrapped_state
        return self.normalize_q(wrapped_state)

    def dsdt(self, state, control, disturbance):
        qw = state[..., 3] * 1.0
        qx = state[..., 4] * 1.0
        qy = state[..., 5] * 1.0
        qz = state[..., 6] * 1.0
        vx = state[..., 7] * 1.0
        vy = state[..., 8] * 1.0
        vz = state[..., 9] * 1.0
        wx = state[..., 10] * 1.0
        wy = state[..., 11] * 1.0
        wz = state[..., 12] * 1.0
        f = (control[..., 0]) * 1.0

        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = vx
        dsdt[..., 1] = vy
        dsdt[..., 2] = vz
        dsdt[..., 3] = -(wx * qx + wy * qy + wz * qz) / 2.0
        dsdt[..., 4] = (wx * qw + wz * qy - wy * qz) / 2.0
        dsdt[..., 5] = (wy * qw - wz * qx + wx * qz) / 2.0
        dsdt[..., 6] = (wz * qw + wy * qx - wx * qy) / 2.0
        k1 = 2 * self.CT / self.m

        dsdt[..., 7] = k1 * self.lin_ctrl_pt[0] * qy  
        dsdt[..., 8] = - k1 * self.lin_ctrl_pt[0] * qx 
        dsdt[..., 9] = self.Gz + 0.5* k1 * f
        dsdt[..., 10] = control[..., 1] * 1.0 
        dsdt[..., 11] = control[..., 2] * 1.0 
        dsdt[..., 12] = control[..., 3] * 1.0

        return dsdt

    def boundary_fn(self, state):
        '''for cylinder with point-mass collision'''
        dist = torch.norm(state[..., :2], dim=-1)
        return torch.maximum(dist, torch.zeros_like(dist)) - self.collisionR
        

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError

        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0
            vx = state[..., 7] * 1.0
            vy = state[..., 8] * 1.0
            vz = state[..., 9] * 1.0
            wx = state[..., 10] * 1.0
            wy = state[..., 11] * 1.0
            wz = state[..., 12] * 1.0

            k1 = 2 * self.CT / self.m

            # Compute the hamiltonian using the simplified linear dynamics under current linearization point
            ham = dvds[..., 0] * vx + dvds[..., 1] * vy + dvds[..., 2] * vz
            # ham += dvds[..., 3] *(-0.5* (qx*wx+qy*wy+qz*wz)/torch.sqrt(torch.clamp(1-qx**2-qy**2-qz**2,min=1e-3)))
            # ham += dvds[..., 4] * 0.5 * wx 
            # ham += dvds[..., 5] * 0.5 * wy 
            # ham += dvds[..., 6] * 0.5 * wz 
            ham += -dvds[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
            ham += dvds[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
            ham += dvds[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
            ham += dvds[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0

            # ham += dvds[..., 7] * k1 * self.lin_ctrl_pt[0] * qy 
            # ham += dvds[..., 8] * (-k1) * self.lin_ctrl_pt[0] * qx 
            # ham += torch.abs(dvds[..., 9]* 0.5* k1 ) * self.collective_thrust_max
            ham += dvds[..., 9] * self.Gz 
            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            ham += torch.abs(dvds[..., 7] * c1 + dvds[..., 8] *
                             c2 +dvds[..., 9]* 0.5* k1 ) * self.collective_thrust_max

            ham += torch.abs(dvds[..., 10]) * self.dwx_max + torch.abs(
                dvds[..., 11]) * self.dwy_max + torch.abs(dvds[..., 12]) * self.dwz_max

            return ham

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError
        elif self.set_mode == 'avoid':

            # u1 = self.collective_thrust_max * torch.sign(dvds[..., 9])
            u1 = -self.Gz * torch.sign(torch.abs(dvds[..., 9]))
            u2 = self.dwx_max * torch.sign(dvds[..., 10])
            u3 = self.dwy_max * torch.sign(dvds[..., 11])
            u4 = self.dwz_max * torch.sign(dvds[..., 12])

        return torch.cat((u1[..., None], u2[..., None], u3[..., None], u4[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return torch.zeros(1)


    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'state_labels': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 7,
        }
    
class QuadrotorLinear(Dynamics):
    def __init__(self, collisionR: float, collective_thrust_max: float,  set_mode: str, N: int):  # simpler quadrotor
        self.collective_thrust_max = collective_thrust_max
        # self.body_rate_acc_max = body_rate_acc_max
        self.m = 1  # mass
        self.arm_l = 0.17
        self.CT = 1
        self.CM = 0.016
        self.Gz = -9.8

        self.dwx_max = 8
        self.dwy_max = 8
        self.dwz_max = 4
        self.dist_dwx_max = 0
        self.dist_dwy_max = 0
        self.dist_dwz_max = 0
        self.dist_f = 0
        self.N= 13

        # self.lin_pt = torch.tensor([0., 0., 0., 0.9848, 0.1, 0.1, 0.1, 0., 0., 0., 0.1, 0.1, 0.1])
        # self.lin_ctrl_pt = torch.tensor([self.m * self.Gz / self.CT, 0.1, 0.1, 0.1])
        self.lin_pt = torch.tensor([0., 0., 0., 1.0, 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.lin_ctrl_pt = torch.tensor([self.collective_thrust_max, 0., 0., 0.])
        # beware, w/ above lin pt, vx and vy are not directly controllable
        # and require a non-zero thrust in lin control pt to be dynamic

        self.collisionR = collisionR

        super().__init__(
            name='QuadrotorLinear', loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=13, input_dim=14, control_dim=4, disturbance_dim=0,
            state_mean=[0 for i in range(13)],
            state_var=[3.0, 3.0, 3.0, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5],
            value_mean=(math.sqrt(3.0**2 + 3.0**2) -
                        2 * self.collisionR) / 2,
            value_var=math.sqrt(3.0**2 + 3.0**2),
            value_normto=0.02,
            deepreach_model="exact"
        )
    def normalize_q(self, x):
        # normalize quaternion
        normalized_x = x*1.0
        q_tensor = x[..., 3:7]*1.0
        q_tensor = torch.nn.functional.normalize(
            q_tensor, p=2,dim=-1)  # normalize quaternion
        normalized_x[..., 3:7] = q_tensor*1.0

        return normalized_x
    
    def control_range(self, state):
        return [[-self.collective_thrust_max, self.collective_thrust_max],
                [-self.dwx_max, self.dwx_max],
                [-self.dwy_max, self.dwy_max],
                [-self.dwz_max, self.dwz_max]]

    def state_test_range(self):
        return [
            [-3.0, 3.0],
            [-3.0, 3.0],
            [-3.0, 3.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # return wrapped_state
        return self.normalize_q(wrapped_state)

    def dsdt(self, state, control, disturbance):
        qw = state[..., 3] * 1.0 #x4, x[3]
        qx = state[..., 4] * 1.0 #x5, x[4]
        qy = state[..., 5] * 1.0 #x6, x[5]
        qz = state[..., 6] * 1.0 #x7, x[6]
        vx = state[..., 7] * 1.0 #x8, x[7]
        vy = state[..., 8] * 1.0 #x9, x[8]
        vz = state[..., 9] * 1.0 #x10, x[9]
        wx = state[..., 10] * 1.0 #x11, x[10]
        wy = state[..., 11] * 1.0 #x12, x[11]
        wz = state[..., 12] * 1.0 #x13, x[12]
        f = (control[..., 0]) * 1.0 #u1, u[0]

        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = vx
        dsdt[..., 1] = vy
        dsdt[..., 2] = vz

        # dsdt[..., 3] = -0.5 * (self.lin_pt[10] * qx + self.lin_pt[11] * qy + self.lin_pt[12] * qz +\
        #                        self.lin_pt[4]  * wx + self.lin_pt[5]  * wy + self.lin_pt[6]  * wz)
        
        # dsdt[..., 4] =  0.5 * (self.lin_pt[10] * qw + self.lin_pt[12] * qy - self.lin_pt[11] * qz +\
        #                        self.lin_pt[3]  * wx + self.lin_pt[5]  * wz - self.lin_pt[6]  * wy)
        
        # dsdt[..., 5] =  0.5 * (self.lin_pt[11] * qw - self.lin_pt[12] * qx + self.lin_pt[10] * qz +\
        #                        self.lin_pt[3]  * wy - self.lin_pt[4]  * wz + self.lin_pt[6]  * wx)
        
        # dsdt[..., 6] =  0.5 * (self.lin_pt[12] * qw + self.lin_pt[11] * qx - self.lin_pt[10] * qy +\
        #                        self.lin_pt[3]  * wz + self.lin_pt[4]  * wy - self.lin_pt[5]  * wx)
        dsdt[..., 3] = -(wx * qx + wy * qy + wz * qz) / 2.0
        dsdt[..., 4] = (wx * qw + wz * qy - wy * qz) / 2.0
        dsdt[..., 5] = (wy * qw - wz * qx + wx * qz) / 2.0
        dsdt[..., 6] = (wz * qw + wy * qx - wx * qy) / 2.0

        k1 = 2 * self.CT / self.m

        dsdt[..., 7] = k1 * self.lin_ctrl_pt[0] * (self.lin_pt[5] * qw + self.lin_pt[6] * qx +\
                                                   self.lin_pt[3] * qy + self.lin_pt[4] * qz) +\
                        k1 * (self.lin_pt[3] * self.lin_pt[5] + self.lin_pt[4] * self.lin_pt[6]) * f
        
        dsdt[..., 8] = k1 * self.lin_ctrl_pt[0] * (-self.lin_pt[4] * qw + self.lin_pt[6] * qy -\
                                                self.lin_pt[3] * qx + self.lin_pt[5] * qz) +\
                        k1 * (-self.lin_pt[3] * self.lin_pt[4] + self.lin_pt[5] * self.lin_pt[6]) * f

        # dsdt[..., 9] = self.Gz - k1 * self.lin_ctrl_pt[0] * (2 * self.lin_pt[4] * qx + 2 * self.lin_pt[5] * qy - 0.5) +\
        #                - k1 * (self.lin_pt[4] * self.lin_pt[4] + self.lin_pt[5] * self.lin_pt[5] - 0.5) * f
        dsdt[..., 9] = self.Gz - k1 * self.lin_ctrl_pt[0] * (2 * self.lin_pt[4] * qx + 2 * self.lin_pt[5] * qy) +\
                       - k1 * (self.lin_pt[4] * self.lin_pt[4] + self.lin_pt[5] * self.lin_pt[5] - 0.5) * f
        
        k2 = 5./9.
        dsdt[..., 10] = (control[..., 1]
                         ) * 1.0 - k2 * (self.lin_pt[11] * wz + self.lin_pt[12] * wy)
        dsdt[..., 11] = (control[..., 2]
                         ) * 1.0 + k2 * (self.lin_pt[10] * wz + self.lin_pt[12] * wx)
        dsdt[..., 12] = (control[..., 3]) * 1.0

        # # Simplified dynamics under the current lin_pt
        # # dsdt[..., 3] = 0
        # # dsdt[..., 3] = -0.5* (qx*wx+qy*wy+qz*wz)/torch.sqrt(torch.clamp(1-qx**2-qy**2-qz**2,min=1e-3))
        # # dsdt[..., 4] =  0.5 * wx 
        # # dsdt[..., 5] =  0.5 * wy 
        # # dsdt[..., 6] =  0.5 * wz 
        # dsdt[..., 3] = -(wx * qx + wy * qy + wz * qz) / 2.0
        # dsdt[..., 4] = (wx * qw + wz * qy - wy * qz) / 2.0
        # dsdt[..., 5] = (wy * qw - wz * qx + wx * qz) / 2.0
        # dsdt[..., 6] = (wz * qw + wy * qx - wx * qy) / 2.0
        # k1 = 2 * self.CT / self.m

        # dsdt[..., 7] = k1 * self.lin_ctrl_pt[0] * qy  
        # dsdt[..., 8] = - k1 * self.lin_ctrl_pt[0] * qx 
        # dsdt[..., 9] = self.Gz + 0.5* k1 * f
        # dsdt[..., 10] = control[..., 1] * 1.0 
        # dsdt[..., 11] = control[..., 2] * 1.0 
        # dsdt[..., 12] = control[..., 3] * 1.0

        return dsdt
    

    ''' non linear dsdt'''
    # def dsdt(self, state, control, disturbance):
    #     qw = state[..., 3] * 1.0
    #     qx = state[..., 4] * 1.0
    #     qy = state[..., 5] * 1.0
    #     qz = state[..., 6] * 1.0
    #     vx = state[..., 7] * 1.0
    #     vy = state[..., 8] * 1.0
    #     vz = state[..., 9] * 1.0
    #     wx = state[..., 10] * 1.0
    #     wy = state[..., 11] * 1.0
    #     wz = state[..., 12] * 1.0
    #     f = (control[..., 0]) * 1.0

    #     dsdt = torch.zeros_like(state)
    #     dsdt[..., 0] = vx
    #     dsdt[..., 1] = vy
    #     dsdt[..., 2] = vz
    #     dsdt[..., 3] = -(wx * qx + wy * qy + wz * qz) / 2.0
    #     dsdt[..., 4] = (wx * qw + wz * qy - wy * qz) / 2.0
    #     dsdt[..., 5] = (wy * qw - wz * qx + wx * qz) / 2.0
    #     dsdt[..., 6] = (wz * qw + wy * qx - wx * qy) / 2.0
    #     dsdt[..., 7] = 2 * (qw * qy + qx * qz) * self.CT / \
    #         self.m * f
    #     dsdt[..., 8] = 2 * (-qw * qx + qy * qz) * self.CT / \
    #         self.m * f
    #     dsdt[..., 9] = self.Gz + (1 - 2 * torch.pow(qx, 2) - 2 *
    #                               torch.pow(qy, 2)) * self.CT / self.m * f
    #     dsdt[..., 10] = (control[..., 1]
    #                      ) * 1.0 - 5 * wy * wz / 9.0
    #     dsdt[..., 11] = (control[..., 2]
    #                      ) * 1.0 + 5 * wx * wz / 9.0
    #     dsdt[..., 12] = (control[..., 3]) * 1.0

    #     return dsdt
    
    def boundary_fn(self, state):
        '''for cylinder with full body collision'''
        # create normal vector
        # v = torch.zeros_like(state[..., 4:7])
        # v[..., 2] = 1
        # v = quaternion.quaternion_apply(state[..., 3:7], v)
        # vx = v[..., 0]
        # vy = v[..., 1]
        # vz = v[..., 2]
        # # compute vector from center of quadrotor to the center of cylinder
        # px = state[..., 0]
        # py = state[..., 1]

        # # get full body distance
        # dist = torch.norm(state[..., :2], dim=-1)
        # # return dist- self.collisionR
        # dist = dist- torch.sqrt((self.arm_l**2*px**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2)
        #                    + (self.arm_l**2*py**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2))
        # return torch.maximum(dist, torch.zeros_like(dist)) - self.collisionR
        
        dist = torch.norm(state[..., :2], dim=-1)
        return torch.maximum(dist, torch.zeros_like(dist)) - self.collisionR

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError

        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0
            vx = state[..., 7] * 1.0
            vy = state[..., 8] * 1.0
            vz = state[..., 9] * 1.0
            wx = state[..., 10] * 1.0
            wy = state[..., 11] * 1.0
            wz = state[..., 12] * 1.0

            # c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            # c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            # c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
            #       torch.pow(qy, 2)) * self.CT / self.m

            k1 = 2 * self.CT / self.m
            c1 = k1 * (self.lin_pt[3] * self.lin_pt[5] + self.lin_pt[4] * self.lin_pt[6])
            c2 = k1 * (-self.lin_pt[3] * self.lin_pt[4] + self.lin_pt[5] * self.lin_pt[6])
            c3 = -k1 * (self.lin_pt[4] * self.lin_pt[4] + self.lin_pt[5] * self.lin_pt[5] - 0.5)

            # # Compute the hamiltonian using the simplified linear dynamics under current linearization point
            # ham = dvds[..., 0] * vx + dvds[..., 1] * vy + dvds[..., 2] * vz
            # # ham += dvds[..., 3] *(-0.5* (qx*wx+qy*wy+qz*wz)/torch.sqrt(torch.clamp(1-qx**2-qy**2-qz**2,min=1e-3)))
            # # ham += dvds[..., 4] * 0.5 * wx 
            # # ham += dvds[..., 5] * 0.5 * wy 
            # # ham += dvds[..., 6] * 0.5 * wz 
            # ham += -dvds[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
            # ham += dvds[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
            # ham += dvds[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
            # ham += dvds[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0

            # ham += dvds[..., 7] * k1 * self.lin_ctrl_pt[0] * qy  
            # ham += dvds[..., 8] * (-k1) * self.lin_ctrl_pt[0] * qx 
            # ham += dvds[..., 9] *self.Gz 
            
            # ham += torch.abs(dvds[..., 9]* 0.5* k1 ) * self.collective_thrust_max

            # ham += torch.abs(dvds[..., 10]) * self.dwx_max + torch.abs(
            #     dvds[..., 11]) * self.dwy_max + torch.abs(dvds[..., 12]) * self.dwz_max


            # # Compute the hamiltonian
            ham = dvds[..., 0] * vx + dvds[..., 1] * vy + dvds[..., 2] * vz

            ham += -dvds[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
            ham += dvds[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
            ham += dvds[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
            ham += dvds[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0

            # ham += dvds[..., 4] * 0.5 * (self.lin_pt[10] * qw + self.lin_pt[12] * qy - self.lin_pt[11] * qz +\
            #                    self.lin_pt[3]  * wx + self.lin_pt[5]  * wz - self.lin_pt[6]  * wy)
            
            # ham += dvds[..., 5] * 0.5 * (self.lin_pt[11] * qw - self.lin_pt[12] * qx + self.lin_pt[10] * qz +\
            #                    self.lin_pt[3]  * wy - self.lin_pt[4]  * wz + self.lin_pt[6]  * wx)

            # ham += dvds[..., 6] * 0.5 * (self.lin_pt[12] * qw + self.lin_pt[11] * qx - self.lin_pt[10] * qy +\
            #                    self.lin_pt[3]  * wz + self.lin_pt[4]  * wy - self.lin_pt[5]  * wx)

            ham += dvds[..., 7] * k1 * self.lin_ctrl_pt[0] * (self.lin_pt[5] * qw + self.lin_pt[6] * qx +\
                                                   self.lin_pt[3] * qy + self.lin_pt[4] * qz)
            
            ham += dvds[..., 8] * k1 * self.lin_ctrl_pt[0] * (-self.lin_pt[4] * qw + self.lin_pt[6] * qy -\
                                                    self.lin_pt[3] * qx + self.lin_pt[5] * qz)
            
            # ham += dvds[..., 9] * (self.Gz - k1 * self.lin_ctrl_pt[0] * (2 * self.lin_pt[4] * qx + 2 * self.lin_pt[5] * qy - 0.5))
            
            # ham += dvds[..., 9] * (self.Gz - k1 * self.lin_ctrl_pt[0] * (2 * self.lin_pt[4] * qx + 2 * self.lin_pt[5] * qy))
            

            k2 = 5./9.
            ham += k2 * (dvds[..., 10] * -(self.lin_pt[11] * wz + self.lin_pt[12] * wy) + dvds[..., 11] * (self.lin_pt[10] * wz + self.lin_pt[12] * wx))

            # ham += torch.abs(dvds[..., 7] * c1 + dvds[..., 8] *
            #                  c2 + dvds[..., 9] * c3) * self.collective_thrust_max
            # ham += torch.abs(dvds[..., 9]* (-k1) * (self.lin_pt[4] * self.lin_pt[4] + self.lin_pt[5] * self.lin_pt[5] - 0.5) ) * self.collective_thrust_max

            ham += torch.abs(dvds[..., 10]) * self.dwx_max + torch.abs(
                dvds[..., 11]) * self.dwy_max + torch.abs(dvds[..., 12]) * self.dwz_max

            # # ham = ham+ torch.clamp(torch.abs(dvds[..., 7] * c1)* self.collective_thrust_max,-0.1,0.1) + \
            # #         torch.clamp(torch.abs(dvds[..., 8] * c2)* self.collective_thrust_max,-0.1,0.1) + \
            # #         torch.clamp(torch.abs(dvds[..., 9] * c3)* self.collective_thrust_max + dvds[..., 9] * self.Gz,-0.1,0.1)
            # # ham = ham+ torch.clamp(-dvds[..., 10] * 5 * wy * wz /  9.0+ torch.abs(dvds[..., 10]) * (self.dwx_max),-0.1,0.1) + \
            # #           torch.clamp(dvds[..., 11] * 5 * wx * wz / 9.0+ torch.abs( dvds[..., 11]) * (self.dwy_max) ,-0.1,0.1) + \
            # #         torch.clamp(torch.abs(dvds[..., 12]) * (self.dwz_max),-0.1,0.1)

            return ham

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError
        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0

            # c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            # c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            # c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
            #       torch.pow(qy, 2)) * self.CT / self.m
            
            k1 = 2 * self.CT / self.m
            c1 = k1 * (self.lin_pt[3] * self.lin_pt[5] + self.lin_pt[4] * self.lin_pt[6])
            c2 = k1 * (-self.lin_pt[3] * self.lin_pt[4] + self.lin_pt[5] * self.lin_pt[6])
            c3 = -k1 * (self.lin_pt[4] * self.lin_pt[4] + self.lin_pt[5] * self.lin_pt[5] - 0.5)

            u1 = self.collective_thrust_max * \
                torch.sign(dvds[..., 7] * c1 + dvds[..., 8] *
                           c2 + dvds[..., 9] * c3)
            u2 = self.dwx_max * torch.sign(dvds[..., 10])
            u3 = self.dwy_max * torch.sign(dvds[..., 11])
            u4 = self.dwz_max * torch.sign(dvds[..., 12])

        return torch.cat((u1[..., None], u2[..., None], u3[..., None], u4[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return torch.zeros(1)


    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'state_labels': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 7,
        }

class MultiVehicleCollision(Dynamics):
    def __init__(self):
        self.angle_alpha_factor = 1.2
        self.velocity = 0.6
        self.omega_max = 1.1
        self.collisionR = 0.25
        super().__init__(
            name="MultiVehicleCollision",loss_type='brt_hjivi', set_mode='avoid',
            state_dim=9, input_dim=10, control_dim=3, disturbance_dim=0,
            state_mean=[
                0, 0,
                0, 0, 
                0, 0,
                0, 0, 0,
            ],
            state_var=[
                1, 1,
                1, 1,
                1, 1,
                self.angle_alpha_factor*math.pi, self.angle_alpha_factor*math.pi, self.angle_alpha_factor*math.pi,
            ],
            value_mean=0.25,
            value_var=0.5,
            value_normto=0.02,
            deepreach_model="exact"
        )

    def state_test_range(self):
        return [
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi],           
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 6] = (wrapped_state[..., 6] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 7] = (wrapped_state[..., 7] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 8] = (wrapped_state[..., 8] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state
        
    # dynamics (per car)
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity*torch.cos(state[..., 6])
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 6])
        dsdt[..., 2] = self.velocity*torch.cos(state[..., 7])
        dsdt[..., 3] = self.velocity*torch.sin(state[..., 7])
        dsdt[..., 4] = self.velocity*torch.cos(state[..., 8])
        dsdt[..., 5] = self.velocity*torch.sin(state[..., 8])
        dsdt[..., 6] = control[..., 0]
        dsdt[..., 7] = control[..., 1]
        dsdt[..., 8] = control[..., 2]
        return dsdt
    
    def boundary_fn(self, state):
        boundary_values = torch.norm(state[..., 0:2] - state[..., 2:4], dim=-1) - self.collisionR
        for i in range(1, 2):
            boundary_values_current = torch.norm(state[..., 0:2] - state[..., 2*(i+1):2*(i+1)+2], dim=-1) - self.collisionR
            boundary_values = torch.min(boundary_values, boundary_values_current)
        # Collision cost between the evaders themselves
        for i in range(2):
            for j in range(i+1, 2):
                evader1_coords_index = (i+1)*2
                evader2_coords_index = (j+1)*2
                boundary_values_current = torch.norm(state[..., evader1_coords_index:evader1_coords_index+2] - state[..., evader2_coords_index:evader2_coords_index+2], dim=-1) - self.collisionR
                boundary_values = torch.min(boundary_values, boundary_values_current)
        return boundary_values

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, dvds):
        # Compute the hamiltonian for the ego vehicle
        ham = self.velocity*(torch.cos(state[..., 6]) * dvds[..., 0] + torch.sin(state[..., 6]) * dvds[..., 1]) + self.omega_max * torch.abs(dvds[..., 6])
        # Hamiltonian effect due to other vehicles
        ham += self.velocity*(torch.cos(state[..., 7]) * dvds[..., 2] + torch.sin(state[..., 7]) * dvds[..., 3]) + self.omega_max * torch.abs(dvds[..., 7])
        ham += self.velocity*(torch.cos(state[..., 8]) * dvds[..., 4] + torch.sin(state[..., 8]) * dvds[..., 5]) + self.omega_max * torch.abs(dvds[..., 8])
        return ham

    def optimal_control(self, state, dvds):
        return self.omega_max*torch.sign(dvds[..., [6, 7, 8]])

    def optimal_disturbance(self, state, dvds):
        return 0
    
    def plot_config(self):
        return {
            'state_slices': [
                0, 0, 
                -0.4, 0, 
                0.4, 0,
                math.pi/2, math.pi/4, 3*math.pi/4,
            ],
            'state_labels': [
                r'$x_1$', r'$y_1$',
                r'$x_2$', r'$y_2$',
                r'$x_3$', r'$y_3$',
                r'$\theta_1$', r'$\theta_2$', r'$\theta_3$',
            ],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 6,
        }

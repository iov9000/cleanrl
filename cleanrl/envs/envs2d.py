from gymnasium import Env, spaces
import numpy as np
import cv2  
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import random
import torch
from collections import deque

def plot_reward(discriminator, shape=(200,60), tgt_pos=(190,30), plot_value=False, env=None):
    # generate grid w/ two different targets:
    x = np.arange(0,shape[0])
    y = np.arange(0,shape[1])

    grid = np.array(np.meshgrid(y,x))
    point_list = np.reshape(grid, [2,-1]).T
    #point_list = grid

    tgt_pos = np.array(tgt_pos)
    grid_pos1 = np.concatenate([point_list, np.tile(tgt_pos, (point_list.shape[0],1))],1)

    def dist_f(grid_pos):
         return -(np.sqrt((grid_pos[:,0] - grid_pos[:,3])**2 + (grid_pos[:,1]-grid_pos[:,2])**2))
    
    r_gt_1 = dist_f(grid_pos1)
    r_gt_1 = r_gt_1.reshape(shape)

    # plot rewards for both goals
    if env is None:
        with torch.no_grad():
            r_1 = discriminator.get_reward(ob=torch.Tensor(grid_pos1),ac=None)
            if plot_value:
                v_1 = discriminator.get_value(ob=torch.Tensor(grid_pos1),ac=None)
            
            r_1 = r_1.cpu().numpy()
            
    title = 'reward'
    if plot_value:
        r_1 = v_1.cpu().numpy()
        title = 'value'

    fig = plt.figure(figsize=(20,5))
    ax1 = plt.subplot(121)
    plt.pcolormesh(x,y,r_1.reshape(shape).T)
    plt.colorbar()
    plt.scatter(tgt_pos[0], tgt_pos[1], c='r')
    plt.title(title + '_irl')
    ax2 = plt.subplot(122)
    plt.pcolormesh(x,y,r_gt_1.T)
    plt.colorbar()
    plt.scatter(tgt_pos[0], tgt_pos[1], c='r')
    plt.title(title + '_gt')
    
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

    image_from_plot = (data.reshape(fig.canvas.get_width_height()[::-1] + (3,)) / 255.0)
    img = np.swapaxes(image_from_plot, 0, 2)
    img = np.swapaxes(img, 1, 2)

    return img


class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name
    
    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)
    
    def get_position(self):
        return (self.x, self.y)
    
    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y
        
        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)
    
class Agent(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Agent, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("assets/agent.png") / 255.0
        #self.icon = cv2.circle(image, center_coordinates, radius, color, thickness)

        self.icon_w = 5
        self.icon_h = 5
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))

class Target(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Target, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("assets/star.png") / 255.0
        self.icon_w = 4
        self.icon_h = 4
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
        
class MovePoint(Env):
    def __init__(self, waypoints=[(175,175)], discrete_actions=True, random_init=False):
        super(MovePoint, self).__init__()
        
        # Define a 2-D observation space
        self.canvas_shape = (200,200,3)
        self.observation_shape = (4)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape)*200,
                                            dtype = np.float32)
    
        self.discrete_actions = discrete_actions
        # Define an action space ranging from 0 to 4
        if discrete_actions:
            self.action_space = spaces.Discrete(5,)
        else:
            self.action_space = spaces.Box(low = np.zeros((2,)), 
                                            high = np.ones((2,)),
                                            dtype = np.float32)
                        
        # Create a canvas to render the environment images upon 
        self.canvas = np.ones(self.canvas_shape) * 1
        self.canvas_hist = np.zeros(self.canvas_shape)

        # Define elements present inside the environment
        self.elements = []
        
        # Add targets
        self.n_tgt = len(waypoints)
        self.waypoints = waypoints
        self.targets = []
        
        # Maximum episode length
        self.max_time = 1000

        # Permissible area of helicper to be 
        self.y_min = int (self.canvas_shape[0] * 0.1)
        self.x_min = 0
        self.y_max = int (self.canvas_shape[0] * 0.9)
        self.x_max = self.canvas_shape[1]
        self.random_init = random_init
        self.truncated = False
        self.terminated = False
        
    def draw_elements_on_canvas(self):
        # Init the canvas 
        self.canvas = np.ones(self.canvas_shape) * 1

        # Draw the agent on canvas
        for elem in self.elements:
            elem_shape = elem.icon.shape
            x,y = elem.x, elem.y
            self.canvas[y : y + elem_shape[1], x:x + elem_shape[0]] = elem.icon

        agent = self.agent
        agent_shape = self.agent.icon.shape
        self.canvas_hist[agent.y : agent.y + agent_shape[1], agent.x:agent.x + agent_shape[0]] += 1

        # normalize hist canvas
        #self.canvas_hist = self.canvas_hist / np.sum(self.canvas_hist)
        #text = 'Time Left: {} | Rewards: {}'.format(self.time, self.ep_return)

        # Put the info on canvas 
        #self.canvas = cv2.putText(self.canvas, text, (10,20), font, 0.8, (0,0,0), 1, cv2.LINE_AA)

    def reset(self):
        # Flag that marks the termination of an episode
        self.truncated = False
        self.terminated = False

        # Reset the fuel consumed
        self.time = self.max_time

        # Reset the reward
        self.curr_tgt = 0

        # Determine a place to intialise the agent in
        if self.random_init:
            x = random.randrange(int(self.canvas_shape[0] * 0.05), int(self.canvas_shape[0] * 0.10))
            y = random.randrange(int(self.canvas_shape[1] * 0.15), int(self.canvas_shape[1] * 0.20))
        else:
            x = 10
            y = 10

        # Intialise the agent
        self.agent = Agent("agent", self.x_max, self.x_min, self.y_max, self.y_min)
        self.agent.set_position(x,y)

        # Intialise the elements 
        self.elements = [self.agent]
        
        # Set the targets
        # self.targets = self.generate_targets()
        
        # define two targets to simulate different experts
        #pos = [(int(self.canvas_shape[0]/2)+self.shift_x,int(self.canvas_shape[1]/2)+self.shift_y), 
        #       (int(self.canvas_shape[0]*.95),int(self.canvas_shape[1]*.95))]
        #self.targets = []
        for i,p in enumerate(self.waypoints):
            tgt = Target("tgt_{}".format(i), self.x_max, self.x_min, self.y_max, self.y_min)
            tgt.set_position(p[0], p[1])
            self.targets.append(tgt)
            
        self.elements.extend(self.targets)
        
        # Reset the Canvas 
        self.canvas = np.ones(self.canvas_shape) * 1

        # Draw elements on the canvas
        self.draw_elements_on_canvas()
        
        curr_tgt = self.targets[self.curr_tgt]
        state = np.stack([self.agent.x, self.agent.y, curr_tgt.x, curr_tgt.y])
        
        self.needs_reset = False
        info = {}

        # return the observation
        return state, info
    
    # TODO: expand to preferences as random process!
    def generate_random_targets(self):
        tgts = []
        for i in range(self.n_tgt):
            tgt = Target("tgt_{}".format(i), self.x_max, self.x_min, self.y_max, self.y_min)

            tgt_x = random.randrange(self.y_min+int(self.y_max/4), self.y_max-int(self.y_max/4))
            tgt_y = random.randrange(self.y_min+int(self.y_max/4), self.y_max-int(self.y_max/4))
            tgt.set_position(tgt_x, tgt_y)
            tgts.append(tgt)
        
        return tgts
    
    def step(self, action):
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"

        # Decrease the time counter 
        self.time -= 1 

        # apply the action to the agent
        if self.discrete_actions:
            if action == 0:
                self.agent.move(0,2)
            elif action == 1:
                self.agent.move(0,-2)
            elif action == 2:
                self.agent.move(2,0)
            elif action == 3:
                self.agent.move(-2,0)
        else:
            self.agent.move(action[0],action[1])

        # REMOVE NOOP
        #elif action == 4:
        #    self.agent.move(0,0)
            
        curr_tgt = self.targets[self.curr_tgt]
        # l2 distance as reward
        reward = -(np.sqrt((self.agent.x - curr_tgt.x)**2 + (self.agent.y-curr_tgt.y)**2))
        
        if self.has_collided(self.agent, curr_tgt):
            #reward += 5
            if self.curr_tgt == len(self.targets)-1:
                # task solved
                reward += 1000
                self.terminated = True
            else:
                # update target
                reward += 500
                self.curr_tgt+=1

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        state = np.stack([self.agent.x, self.agent.y, curr_tgt.x, curr_tgt.y])
                
        # If out of fuel, end the episode.
        if self.time == 0:
            self.truncated = True
            self.needs_reset = True

        return state, reward, self.terminated, self.truncated, {}
    
    def has_collided(self, elem1, elem2):
        x_col = False
        y_col = False

        elem1_x, elem1_y = elem1.get_position()
        elem2_x, elem2_y = elem2.get_position()

        if 2 * abs(elem1_x - elem2_x) <= (elem1.icon_w + elem2.icon_w):
            x_col = True

        if 2 * abs(elem1_y - elem2_y) <= (elem1.icon_h + elem2.icon_h):
            y_col = True

        if x_col and y_col:
            return True

        return False

    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            heatmapimg = np.array(self.canvas_hist * 255, dtype = np.uint8)
            heatmap = cv2.applyColorMap(heatmapimg, cv2.COLORMAP_JET)
            heatmap = heatmap/255
            cat_img = np.hstack((self.canvas, np.ones([200,4,3])*0.2, heatmap))
            cv2.imshow("game", cat_img)
            #plt.imshow("Game", cat_img)
            cv2.waitKey(10)

        elif mode == "rgb_array":
            return self.canvas

    def close(self):
        cv2.destroyAllWindows()
        plt.close('all')


# discrete movepoint
class MoveP(Env):
    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, init_pos=(10,30), waypoints=[(100, 30), (190, 30)], shape=(200,60), 
                random_init=False, goal_conditioning=True, render_mode="rgb_array"):
        super(MoveP, self).__init__()
        
        self.goal_conditioning = goal_conditioning
        if goal_conditioning:
            dim = 4
        else:
            dim = 2

        # Define a 2-D observation space (R^4)
        self.shape = shape
        self.observation_shape = (dim)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape)*np.max(self.shape),
                                            dtype = np.float32)
    
        self.action_space = spaces.Discrete(5,)

        
        # Add targets
        self.n_tgt = len(waypoints)
        self.waypoints = waypoints
        
        # Maximum episode length
        self.max_time = 1000

        self.init_pos = init_pos
        # Permissible area of helicper to be 
        self.y_min = 0
        self.x_min = 0
        self.x_max = self.shape[0]
        self.y_max = self.shape[1]

        self.random_init = random_init
        
    def reset(self, seed=1, options=None):
        # Flag that marks the termination of an episode
        self.term = False
        self.trunc = False
        # Reset the fuel consumed
        self.time = self.max_time

        # Reset the reward
        self.curr_tgt = 0

        # Determine a place to intialise the agent in
        if self.random_init:
            x_noise = random.randrange(int(self.shape[0] * 0.05), int(self.shape[0] * 0.10))
            y_noise = random.randrange(int(self.shape[1] * 0.15), int(self.shape[1] * 0.20))
            x, y = self.init_pos
            x += x_noise
            y += y_noise
        else:
            x, y = self.init_pos

        # Intialise the agent
        self.agent = np.array([x,y])
        
        curr_tgt = np.array(self.waypoints[self.curr_tgt])
        if self.goal_conditioning:
            state = np.concatenate([self.agent, curr_tgt],0)
        else:
            state = self.agent
        
        self.needs_reset = False
        # return the observation
        return state, {}
    
    # TODO: expand to preferences as random process!
    def generate_random_targets(self):
        tgts = []
        for i in range(self.n_tgt):
            tgt = Target("tgt_{}".format(i), self.x_max, self.x_min, self.y_max, self.y_min)

            tgt_x = random.randrange(self.y_min+int(self.y_max/4), self.y_max-int(self.y_max/4))
            tgt_y = random.randrange(self.y_min+int(self.y_max/4), self.y_max-int(self.y_max/4))
            tgt.set_position(tgt_x, tgt_y)
            tgts.append(tgt)
        
        return tgts
    
    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)

    def get_reward_grid(self):
        x = np.arange(0,self.shape[0])
        y = np.arange(0,self.shape[1])

        grid = np.array(np.meshgrid(y,x))
        point_list = np.reshape(grid, [2,-1]).T

        grids = []
        for wp in self.waypoints:
            if self.goal_conditioning:
                grids.append(np.concatenate([point_list, np.tile(wp, (point_list.shape[0],1))],-1))
            else:
                grids.append(point_list)

        return grids

    def get_reward(self, state, tgt = (175,175)):
        curr_tgt = np.array(tgt)
        # l2 distance as reward
        reward = -(np.sqrt((state[0] - curr_tgt[0])**2 + (state[1]-curr_tgt[1])**2))
        
        return reward

    def step(self, action):
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"

        # Decrease the time counter 
        self.time -= 1 

        # apply the action to the agent

        if action == 0:
            move = (0,2)
        elif action == 1:
            move = (0,-2)
        elif action == 2:
            move = (2,0)
        elif action == 3:
            move = (-2,0)
        else:
            move = (0,0)

        self.agent[0] = self.clamp(self.agent[0] + move[0], self.x_min, self.x_max)
        self.agent[1] = self.clamp(self.agent[1] + move[1], self.y_min, self.y_max)

        # REMOVE NOOP
        #elif action == 4:
        #    self.agent.move(0,0)
            
        curr_tgt = np.array(self.waypoints[self.curr_tgt])
        # l2 distance as reward
        reward = -(np.sqrt((self.agent[0] - curr_tgt[0])**2 + (self.agent[1]-curr_tgt[1])**2))
        
        if np.linalg.norm(self.agent - curr_tgt) < 3:
            #reward += 5
            if self.curr_tgt == len(self.waypoints)-1:
                # task solved
                reward += 1000
                self.term = True
            else:
                # update target
                reward += 500
                self.curr_tgt+=1
        
        if self.goal_conditioning:
            state = np.stack([self.agent[0], self.agent[1], curr_tgt[0], curr_tgt[1]])
        else:
            state = self.agent

        # If out of fuel, end the episode.
        if self.time == 0:
            self.trunc = True
            self.needs_reset = True

        return state, reward, self.term, self.trunc, {}

    def render(self, mode = "rgb_array"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        fig = plt.figure(figsize=(10,10))
        plt.scatter(self.agent[0], self.agent[1], c='m',alpha=0.2)
        curr_tgt = self.waypoints[self.curr_tgt]
        plt.scatter(curr_tgt[0], curr_tgt[1], c='r')
        plt.axis('equal')
        if mode == "human":
            plt.show()

        elif mode == "rgb_array":
            #return fig
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            return data

# continuous movepoint
class CMoveP(Env):
    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, init_pos=(10,30), waypoints=[(100, 30), (190, 30)], shape=(200,60), 
                random_init=False, goal_conditioning=True, render_mode="rgb_array"):
        super(CMoveP, self).__init__()
        
        self.goal_conditioning = goal_conditioning
        if goal_conditioning:
            dim = 4
        else:
            dim = 2

        # Define a 2-D observation space (R^4)
        self.shape = shape
        self.observation_shape = (dim)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape)*np.max(self.shape),
                                            dtype = np.float32)
    
        self.action_space = spaces.Box(low = -3*np.ones((2,)), 
                                            high = 3*np.ones((2,)),
                                            dtype = np.float32)
        
        # Add targets
        self.n_tgt = len(waypoints)
        self.waypoints = waypoints
        
        # Maximum episode length
        self.max_time = 1000

        # Permissible area of agent to be 
        self.y_min = 0
        self.x_min = 0
        self.x_max = self.shape[0]
        self.y_max = self.shape[1]

        self.random_init = random_init
        
    def reset(self, seed=1, options=None):
        # Flag that marks the termination of an episode
        self.term = False
        self.trunc = False
        # Reset the fuel consumed
        self.time = self.max_time

        # Reset the reward
        self.curr_tgt = 0

        # Determine a place to intialise the agent in
        if self.random_init:
            x_noise = random.randrange(int(self.shape[0] * 0.05), int(self.shape[0] * 0.10))
            y_noise = random.randrange(int(self.shape[1] * 0.15), int(self.shape[1] * 0.20))
        else:
            x = 10
            y = 10

        # Intialise the agent
        self.agent = np.array([x,y])
        
        curr_tgt = np.array(self.waypoints[self.curr_tgt])
        if self.goal_conditioning:
            state = np.concatenate([self.agent, curr_tgt],0)
        else:
            state = self.agent
        
        self.needs_reset = False
        # return the observation
        return state, {}
    
    # TODO: expand to preferences as random process!
    def generate_random_targets(self):
        tgts = []
        for i in range(self.n_tgt):
            tgt = Target("tgt_{}".format(i), self.x_max, self.x_min, self.y_max, self.y_min)

            tgt_x = random.randrange(self.y_min+int(self.y_max/4), self.y_max-int(self.y_max/4))
            tgt_y = random.randrange(self.y_min+int(self.y_max/4), self.y_max-int(self.y_max/4))
            tgt.set_position(tgt_x, tgt_y)
            tgts.append(tgt)
        
        return tgts
    
    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)

    def get_reward(self, state, tgt = (175,175)):
        curr_tgt = np.array(tgt)
        # l2 distance as reward
        reward = -(np.sqrt((state[0] - curr_tgt[0])**2 + (state[1]-curr_tgt[1])**2))
        
        return reward

    def get_reward_grid(self):
        x = np.arange(0,self.shape[0])
        y = np.arange(0,self.shape[1])

        grid = np.array(np.meshgrid(y,x))
        point_list = np.reshape(grid, [2,-1]).T

        grids = []
        for wp in self.waypoints:
            if self.goal_conditioning:
                grids.append(np.concatenate([point_list, np.tile(wp, (point_list.shape[0],1))],-1))
            else:
                grids.append(point_list)

        return grids

    def step(self, action):
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"

        # Decrease the time counter 
        self.time -= 1 

        # apply the action to the agent
        move = action

        self.agent[0] = self.clamp(self.agent[0] + move[0], self.x_min, self.x_max)
        self.agent[1] = self.clamp(self.agent[1] + move[1], self.y_min, self.y_max)

        # REMOVE NOOP
        #elif action == 4:
        #    self.agent.move(0,0)
            
        curr_tgt = np.array(self.waypoints[self.curr_tgt])
        # l2 distance as reward
        reward = -(np.sqrt((self.agent[0] - curr_tgt[0])**2 + (self.agent[1]-curr_tgt[1])**2))
        
        if np.linalg.norm(self.agent - curr_tgt) < 3:
            #reward += 5
            if self.curr_tgt == len(self.waypoints)-1:
                # task solved
                reward += 1000
                self.term = True
            else:
                # update target
                reward += 500
                self.curr_tgt+=1
        
        if self.goal_conditioning:
            state = np.stack([self.agent[0], self.agent[1], curr_tgt[0], curr_tgt[1]])
        else:
            state = self.agent

        # If out of fuel, end the episode.
        if self.time == 0:
            self.trunc = True
            self.needs_reset = True

        return state, reward, self.term, self.trunc, {}

    def render(self, mode = "rgb_array"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        fig = plt.figure(figsize=(10,10))
        plt.scatter(self.agent[0], self.agent[1], c='m',alpha=0.2)
        curr_tgt = self.waypoints[self.curr_tgt]
        plt.scatter(curr_tgt[0], curr_tgt[1], c='r')
        plt.axis('equal')
        if mode == "human":
            plt.show()

        elif mode == "rgb_array":
            #return fig
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            return data


# continuous movepoint with obstacles
class CMovePND(Env):
    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, waypoints=[(198,198)], D=2, size=200, 
            n_obstacles=1, random_obstacles=False, hard_obstacles=False, safety_reward_coeff = 0.0, 
            random_init=False, goal_conditioning=True, fail_on_collision=False, render_mode="rgb_array"):
        super(CMovePND, self).__init__()
        

        self.render_mode = render_mode

        self.dim = D
        self.goal_conditioning = goal_conditioning
        if goal_conditioning:
            dim = 2*D
        else:
            dim = 2

        # Define a 2-D observation space (R^4)
        self.size = size
        self.observation_shape = (dim)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape)*self.size,
                                            dtype = np.float32)
    
        self.action_space = spaces.Box(low = -3*np.ones((D,)), 
                                            high = 3*np.ones((D,)),
                                            dtype = np.float32)
        
        # Permissible area of point to be 
        self.min = np.array([0.]*D)
        self.max = np.array([self.size]*D)
        self.random_init = random_init
        # Maximum episode length
        self.max_time = 1000
        self.safety_reward_coeff = safety_reward_coeff
        self.agent_render_q = deque(maxlen=100)
        self.hard_obstacles = bool(hard_obstacles)
        self.fail_on_collision = bool(fail_on_collision)

        self.collision_flag = False

        # Add targets
        self.n_tgt = len(waypoints)
        self.waypoints = waypoints
        if random_obstacles:
            self.obstacles = self.gen_random_obstacles(n_obstacles)
        else:
            self.obstacles = self.gen_fixed_obstacles([(np.array((size/4, size/8)), np.array((size/5, size/5))), 
                                                       (np.array((size/2, size/8*3)), np.array((size/4, size/4)))])
        
    def gen_random_obstacles(self, n_obstacles):
        self.obstacles = []
        for i in range(n_obstacles):
            x0 = np.clip(np.random.randint(0.3*self.size,0.7*self.size, self.dim),self.min, self.max)
            xlen = np.clip(np.random.randint(0, 0.2*self.size, self.dim), self.min, self.max)
            self.obstacles.append((x0, xlen))
        
        return self.obstacles

    def gen_fixed_obstacles(self, obstacle_specs):
        self.obstacles = []
        for (x0,xlen) in obstacle_specs:
            self.obstacles.append((x0, xlen))
        
        return self.obstacles
        
    def check_obstacles(self, state, action):
        res = state+action
        safety_rew = 0
        for i, j in self.obstacles:
            mask0 = state+action > i
            mask1 = state+action < i+j

            if np.all(mask0) and np.all(mask1):
                if self.hard_obstacles:
                    res = state
                    
                safety_rew = -1

            #mask = np.logical_or(mask0, mask1)
            #res = mask*(state+action) + np.logical_not(mask)*state
            
        return res, safety_rew

    def reset(self, seed=1, options=None):
        # Flag that marks the termination of an episode
        self.term = False
        self.trunc = False
        # Reset the fuel consumed
        self.time = self.max_time

        # Reset the reward
        self.curr_tgt = 0

        # Determine a place to intialise the agent in
        if self.random_init:
            x = np.random.rand(self.dim)*self.size*0.1
        else:
            x = np.zeros(self.dim)

        # Intialise the agent
        self.agent = x
        
        curr_tgt = np.array(self.waypoints[self.curr_tgt])
        state = np.concatenate([self.agent, curr_tgt],0)
        
        self.state = state

        self.needs_reset = False
        # return the observation
        return state, {}

    def get_state(self):
        return self.state, self.agent, self.curr_tgt

    def set_state(self, agent, tgt):
        self.agent = agent
        self.curr_tgt = tgt
        curr_tgt = np.array(self.waypoints[self.curr_tgt])
        self.state = np.concatenate([self.agent, curr_tgt],0)

    def step(self, action):
        # Assert that it is a valid action 
        #assert self.action_space.contains(action), "Invalid Action"

        # Decrease the time counter 
        self.time -= 1 

        # apply the action to the agent
        move = action

        next_state, safety_rew = self.check_obstacles(self.agent, move)
        if safety_rew < 0:
            self.collision_flag = True
        else:
            self.collision_flag = False

        self.agent = self.clamp(next_state, self.min, self.max)
        #self.agent = self.clamp(self.agent + move, self.min, self.max)

        # compute obstacles
        
        # REMOVE NOOP
        #elif action == 4:
        #    self.agent.move(0,0)
            
        curr_tgt = np.array(self.waypoints[self.curr_tgt])
        # l2 distance as reward
        if self.safety_reward_coeff > 0.0:
            reward = self.get_reward(self.agent, curr_tgt, 
                            safety_rew=self.safety_reward_coeff*safety_rew)
            
            if self.fail_on_collision and self.collision_flag:
                self.term = True
                reward -= self.safety_reward_coeff
        else:
            reward = self.get_reward(self.agent, curr_tgt)

        if np.linalg.norm(self.agent - curr_tgt) < 3:
            #reward += 5
            if self.curr_tgt == len(self.waypoints)-1:
                # task solved
                reward += 1000
                self.term = True
            else:
                # update target
                reward += 500
                self.curr_tgt+=1
        
        if self.goal_conditioning:
            state = np.concatenate([self.agent, curr_tgt], -1)
        else:
            state = self.agent

        self.state = state

        # If out of fuel, end the episode.
        if self.time == 0:
            self.trunc = True
            self.needs_reset = True

        return state, reward, self.term, self.trunc, {}
                        
    def clamp(self, n, minn, maxn):
        return np.clip(n, minn, maxn)

    def get_reward(self, state, tgt = (175,175), safety_rew = 0):
        curr_tgt = np.array(tgt)
        # l2 distance as reward
        reward = -np.linalg.norm(state-tgt) + safety_rew
        
        return reward
                            
    def render(self, mode = "rgb_array"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        fig = plt.figure(figsize=(8,8), dpi=200)
        ax = plt.gca()


        ## Plot agent
        self.agent_render_q.append((self.agent, self.collision_flag))
        for i, (agent, collision) in enumerate(self.agent_render_q):
            ax.plot(agent[0], agent[1], c='red')
            if collision:
                marker = '*'
                color = 'red'
            else:
                marker = '.'
                color = 'green'
            ax.scatter(agent[0], agent[1], s=20, c=color, marker=marker, alpha=(i+1)/(len(self.agent_render_q)+1))

        curr_tgt = self.waypoints[self.curr_tgt]
        plt.scatter(curr_tgt[0], curr_tgt[1], c='k', s=10, marker='*')
        plt.axis('equal')


        ax.set_ylim(0,self.size)
        ax.set_xlim(0,self.size)
        #ax.set_xticks([0, self.size/2, self.size])
        #ax.set_yticks([0, self.size/2, self.size])
        dim0, dim1 = 0,1
        for i, j in self.obstacles:
            rect = patches.Rectangle((i[dim0],i[dim1]), j[dim0], j[dim1], linewidth=2, edgecolor='m', facecolor='none')
            ax.add_patch(rect)

        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            #return fig
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            plt.close()
            return data[:,:,:3]


def plot_reward(shape=(200,60), tgt_pos=(190,30), plot_value=False, env=None):
    # generate grid w/ two different targets:
    x = np.arange(0,shape[0])
    y = np.arange(0,shape[1])

    grid = np.array(np.meshgrid(y,x))
    point_list = np.reshape(grid, [2,-1]).T
    #point_list = grid

    tgt_pos = np.array(tgt_pos)
    grid_pos1 = np.concatenate([point_list, np.tile(tgt_pos, (point_list.shape[0],1))],1)

    def dist_f(grid_pos):
        return -(np.sqrt((grid_pos[:,0] - grid_pos[:,3])**2 + (grid_pos[:,1]-grid_pos[:,2])**2))

    r_gt_1 = dist_f(grid_pos1)
    r_gt_1 = r_gt_1.reshape(np.array(shape))

    title = 'reward'

    fig = plt.figure(figsize=(20,5))
    ax1 = plt.subplot(121)
    plt.pcolormesh(x, y, r_gt_1.T)
    plt.colorbar()
    plt.scatter(tgt_pos[0], tgt_pos[1], c='r')
    plt.title(title + '_irl')
    ax2 = plt.subplot(122)
    plt.pcolor(x, y, r_gt_1.T)
    plt.colorbar()
    plt.scatter(tgt_pos[0], tgt_pos[1], c='r')
    plt.title(title + '_gt')
    

    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

    image_from_plot = (data.reshape(fig.canvas.get_width_height()[::-1] + (3,)) / 255.0)
    img = np.swapaxes(image_from_plot, 0, 2)
    img = np.swapaxes(img, 1, 2)

    return img
          
# # continuous movepoint
# class CMoveP(Env):
#     metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 10}

#     def __init__(self, waypoints=[(175,175)], size=200, random_init=False, render_mode="rgb_array"):
#         super(CMoveP, self).__init__()
        
#         # Define a 2-D observation space (R^4)
#         self.size = size
#         self.observation_shape = (4)
#         self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
#                                             high = np.ones(self.observation_shape)*self.size,
#                                             dtype = np.float32)
    
#         self.action_space = spaces.Box(low = -np.ones((2,)), 
#                                             high = np.ones((2,)),
#                                             dtype = np.float32)
        
#         # Add targets
#         self.n_tgt = len(waypoints)
#         self.waypoints = waypoints
        
#         # Maximum episode length
#         self.max_time = 1000

#         # Permissible area of helicper to be 
#         self.y_min = 0
#         self.x_min = 0
#         self.y_max = self.size
#         self.x_max = self.size
#         self.random_init = random_init
        
#     def reset(self, seed=1, options=None):
#         # Flag that marks the termination of an episode
#         self.term = False
#         self.trunc = False

#         # Reset the fuel consumed
#         self.time = self.max_time

#         # Reset the reward
#         self.curr_tgt = 0

#         # Determine a place to intialise the agent in
#         if self.random_init:
#             x = random.randrange(int(self.size * 0.05), int(self.size * 0.10))
#             y = random.randrange(int(self.size * 0.15), int(self.size * 0.20))
#         else:
#             x = 10
#             y = 10

#         # Intialise the agent
#         self.agent = np.array([x,y])
        
#         curr_tgt = np.array(self.waypoints[self.curr_tgt])
#         state = np.concatenate([self.agent, curr_tgt],0)
        
#         self.needs_reset = False

#         self.fig = plt.figure(figsize=(10,10))
#         # return the observation
#         return state, {}
    
#     # TODO: expand to preferences as random process!
#     def generate_random_targets(self):
#         tgts = []
#         for i in range(self.n_tgt):
#             tgt = Target("tgt_{}".format(i), self.x_max, self.x_min, self.y_max, self.y_min)

#             tgt_x = random.randrange(self.y_min+int(self.y_max/4), self.y_max-int(self.y_max/4))
#             tgt_y = random.randrange(self.y_min+int(self.y_max/4), self.y_max-int(self.y_max/4))
#             tgt.set_position(tgt_x, tgt_y)
#             tgts.append(tgt)
        
#         return tgts
    
#     def clamp(self, n, minn, maxn):
#         return max(min(maxn, n), minn)

#     def step(self, action):
#         # Assert that it is a valid action 
#         assert self.action_space.contains(action), "Invalid Action"

#         # Decrease the time counter 
#         self.time -= 1 

#         # apply the action to the agent
#         self.agent[0] = self.clamp(self.agent[0] + action[0], self.x_min, self.x_max)
#         self.agent[1] = self.clamp(self.agent[1] + action[1], self.y_min, self.y_max)


#         # REMOVE NOOP
#         #elif action == 4:
#         #    self.agent.move(0,0)
            
#         curr_tgt = np.array(self.waypoints[self.curr_tgt])
#         # l2 distance as reward
#         reward = -(np.sqrt((self.agent[0] - curr_tgt[0])**2 + (self.agent[1]-curr_tgt[1])**2))
        
#         if np.linalg.norm(self.agent - curr_tgt) < 1e-1:
#             #reward += 5
#             if self.curr_tgt == len(self.waypoints)-1:
#                 # task solved
#                 print("Task Solved")
#                 reward += 1000
#                 self.term = True
#             else:
#                 # update target
#                 reward += 500
#                 self.curr_tgt+=1

#         state = np.stack([self.agent[0], self.agent[1], curr_tgt[0], curr_tgt[1]])
        
#         # If out of fuel, end the episode.
#         if self.time == 0:
#             self.trunc = True
#             self.needs_reset = True

#         return state, reward, self.term, self.trunc, {}

#     def render(self, mode = "rgb_array"):
#         assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
#         plt.gcf()
#         plt.scatter(self.agent[0], self.agent[1], c='m', s=2, alpha=0.2)
#         curr_tgt = self.waypoints[self.curr_tgt]
#         plt.scatter(curr_tgt[0], curr_tgt[1], c='r')
#         plt.axis('equal')
#         if mode == "human":
#             plt.show()

#         elif mode == "rgb_array":
#             self.fig.canvas.draw()

#             data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
#             data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
#             return data


    

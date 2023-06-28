import os
import sys

from gym import Env, spaces
#import open3d as o3d
import trimesh

import numpy as np
import cv2  
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

def load_meshes_o3d(folder, simplify_mesh=False):
    meshes = {}
    vertices = []
    for f in os.listdir(folder):
        if 'obj' in f:
            #meshes[f] = trimesh.load_mesh(os.path.join(folder,f))
            mesh = o3d.io.read_triangle_mesh(os.path.join(folder,f))
            if simplify_mesh:
                voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 32
                mesh_smp = mesh.simplify_vertex_clustering(
                    voxel_size=voxel_size,
                    contraction=o3d.geometry.SimplificationContraction.Average)
                meshes[f] = mesh_smp
            else:
                meshes[f] = mesh
            
            vertices.append(np.array(mesh.vertices))
        
    vertices = np.concatenate(vertices,0)

    belly_center = np.array(meshes['Belly.obj'].get_oriented_bounding_box().get_center())
    targets = []
    for k,v in meshes.items():
        bb_c = np.array(v.get_oriented_bounding_box().get_center())
        tgt = bb_c + (belly_center-bb_c)*0.5
        targets.append(tgt)
    
    return meshes, vertices, targets

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def load_meshes_tri(folder, anatomy_list, simplify_mesh=False, scale=1000):
    meshes = {}
    vertices = []

    for f in os.listdir(folder):
        if 'obj' in f:
            mesh = as_mesh(trimesh.load_mesh(os.path.join(folder,f)))
            mesh.apply_scale(scale)
            if simplify_mesh and 'Belly' not in f:
                mesh = mesh.simplify_quadratic_decimation(2000)
            meshes[f] = mesh
            vertices.append(np.array(mesh.vertices))

    vertices = np.concatenate(vertices,0)

    belly_center = np.array(meshes['Belly.obj'].centroid)
    targets = []
    for k in anatomy_list:
        bb_c = np.array(meshes[k].centroid)
        tgt = bb_c + (belly_center-bb_c)*0.7
        # (sanity) check if target is inside mesh 
        #print(k)
        #print(meshes[k].contains(np.expand_dims(tgt,0)))
        targets.append(tgt)
    
    return meshes, vertices, targets



class Laparos(Env):
    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, step_size=1, mesh_folder='envs/obj', state_spec='r3', gc=False, random_init=False):
        super(Laparos, self).__init__()

        self.anatomy_list = ['DomeOfBladder.obj', 'Uterus.obj', 'CecumAppendix.obj', 'ColonRight.obj', 
                    'LiverRight.obj', 'Falciform.obj', 'LiverLeft.obj', 'GallBladder.obj', 
                    'HepatoDuodenalLigament.obj', 'Spleen.obj', 'Stomach.obj']
        # load meshes        
        self.meshes, self.vertices, self.waypoints = load_meshes_tri(mesh_folder, self.anatomy_list)
        self.n_tgt = len(self.waypoints)
        #self.initial_position = np.array(self.meshes['DomeOfBladder.obj'].get_center()) + np.array([0.05,0.05,0.05]) 
        self.initial_position = np.array(self.meshes['Belly.obj'].centroid)
        self.radius = step_size
        #belly_bbox = np.array(self.meshes['Belly.obj'].get_oriented_bounding_box().get_box_points())
        belly_bbox = np.array(self.meshes['Belly.obj'].bounding_box_oriented.vertices)
        self.space_min = np.min(belly_bbox)
        self.space_max = np.max(belly_bbox)
        self.goal_conditioning = gc
        self.plot_meshes = True

        # Define a 3-D or 7-D observation space 
        if state_spec == 'r3so3':
            obs_dim = 7
        else:
            obs_dim = 3

        if self.goal_conditioning:
            obs_dim = obs_dim*2
        self.observation_shape = (obs_dim,)

        self.observation_space = spaces.Box(low = np.ones(self.observation_shape)*self.space_min, 
                                            high = np.ones(self.observation_shape)*self.space_max,
                                            dtype = np.float32)
    
        if state_spec == 'r3so3':
            self.action_space = spaces.Discrete(11,)
        else:
            self.action_space = spaces.Discrete(7,)
        
        # Define target points based on geometry
        
        # Maximum episode length
        self.max_time = 5000

        self.step_size = step_size
        self.random_init = random_init
        
    def compute_endoscope_movement(state, action):
        # rotate 
        #this.transform.Rotate(this.transform.right, rotation_output * Time.deltaTime, Space.World);
        # Apply actions to transform: translation along forward direction
        #this.transform.Translate(this.transform.right * translation_output * Time.deltaTime, Space.World);

        #Apply actions to transform: pivot around portal
        #.transform.RotateAround(portal.transform.position, artcam.transform.up, pivot_output_x * Time.deltaTime);
        #.transform.RotateAround(portal.transform.position, -artcam.transform.right, pivot_output_y * Time.deltaTime);
        # rotate camera
        return state

    def reset(self):
        # Flag that marks the termination of an episode
        self.done = False
        # Reset the fuel consumed
        self.time = self.max_time

        # Reset the reward
        self.curr_tgt = 0

        # Determine a place to intialise the agent in
        self.agent = self.initial_position
        if self.random_init:
            self.agent + np.random.randn(self.initial_position.shape)*0.001
        
        curr_tgt = np.array(self.waypoints[self.curr_tgt])
        if self.goal_conditioning:
            state = np.concatenate([self.agent, curr_tgt],0)
        else:
            state = self.agent
        
        self.needs_reset = False
        # return the observation
        return state
    
    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)

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
        mag = self.step_size

        if action == 0:
            move = (mag,0,0)
        elif action == 1:
            move = (-mag,0,0)
        elif action == 2:
            move = (0,mag,0)
        elif action == 3:
            move = (0,-mag,0)
        elif action == 4:
            move = (0,0,mag)
        elif action == 5:
            move = (0,0,-mag)
        else:
            move = (0,0,0)

        move_ = np.array(move)

        self.agent = np.clip(self.agent + move_, self.space_min, self.space_max)

        # REMOVE NOOP
        #elif action == 4:
        #    self.agent.move(0,0)
        reward = 0
        # compute distance to meshes and check for collisions
        #dist = (self.agent-self.vertices)**2 < self.radius**2
        for k,v in self.meshes.items():
            dist = np.sum((self.agent-v.vertices)**2,-1) < self.radius
            if np.any(dist):
                print(f"Collision with {k}")
                reward += -100000
                self.done = True
                self.needs_reset = True
            
        curr_tgt = np.array(self.waypoints[self.curr_tgt])
        # l2 distance as reward
        reward += -np.linalg.norm(self.agent - curr_tgt)

        if np.linalg.norm(self.agent - curr_tgt) < 2*self.step_size:
            #reward += 5
            print(f"Task solved: {self.anatomy_list[self.curr_tgt]} reached ")

            if self.curr_tgt == len(self.waypoints)-1:
                # task solved
                reward += 1000
                self.done = True
            else:
                # update target
                reward += 500
                self.curr_tgt+=1

        if self.goal_conditioning:
            state = np.concatenate([self.agent, curr_tgt], -1)
        else:
            state = self.agent
        
        # If out of fuel, end the episode.
        if self.time == 0:
            self.done = True
            self.needs_reset = True

        return state, reward, self.done, {}

    def render(self, mode = "rgb_array"):
        # do 3d plot with meshes
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        matplotlib.style.use('seaborn')

        #fig = plt.gcf()
        fig = plt.figure(figsize=[10,10])
        ax = fig.add_subplot(111, projection='3d')
        #ax = Axes3D(fig)
        ax.view_init(-50, 90)
        ax.dist = 5

        idx = np.arange(500,560)
        scale = 1

        #ax.view_init(-10, 90)
        total_points = 0
        for k,v in self.meshes.items():
            total_points += len(v.vertices)
            vtx = np.array(v.vertices)
            idx = np.arange(int(len(v.vertices)/2)-30, int(len(v.vertices)/2)+30)
            #bb = np.array(v.get_oriented_bounding_box().get_box_points())
            #bb_c = np.array(v.get_oriented_bounding_box().get_center())
            bb = np.array(v.bounding_box_oriented.vertices)*scale
            bb_c = np.array(v.centroid)*scale
            if self.plot_meshes:
                if 'LiverRight' in k:
                    ax.scatter(-vtx[:,0],vtx[:,1],-vtx[:,2], s=0.3)
                    ax.scatter(-vtx[idx,0],vtx[idx,1],-vtx[idx,2], s=3, c='cyan')
                else:
                    ax.scatter(vtx[:,0],vtx[:,1],vtx[:,2], s=0.3)
                    ax.scatter(vtx[idx,0],vtx[idx,1],vtx[idx,2], s=3, c='cyan')

            #ax.scatter(bb[:,0],bb[:,1],bb[:,2], s=10, c='red')
            ax.scatter(bb_c[0],bb_c[1],bb_c[2], s=10, c='yellow')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        # plot agent
        ax.scatter(self.agent[0],self.agent[1],self.agent[2], s=50, c='red')

        wp = np.array(self.waypoints)
        ax.scatter(wp[:,0], wp[:,1],wp[:,2], s=30, c='green')
        wp_curr = wp[self.curr_tgt]

        ax.scatter(wp_curr[0],wp_curr[1],wp_curr[2], s=40, c='m')

        # plot demos?

        if mode == "human":
            plt.show()

        elif mode == "rgb_array":
            #return fig
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            return data

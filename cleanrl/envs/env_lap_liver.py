import os
import sys

from gymnasium import Env, spaces
import trimesh

from collections import deque
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
from sklearn.neighbors import NearestNeighbors


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


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

def load_meshes_tri(folder, anatomy_list, simplify_mesh=False, scale=100):
    meshes = {}
    vertices = []

    for f in os.listdir(folder):
        if 'obj' in f:
            mesh = as_mesh(trimesh.load_mesh(os.path.join(folder,f)))
            mesh.apply_scale(scale)
            if simplify_mesh and 'Belly' not in f:
                mesh = mesh.simplify_quadratic_decimation(2000)
            meshes[f] = mesh
            #if 'Belly' not in f:
            vertices.append(np.array(mesh.vertices))

    vertices = np.concatenate(vertices,0)
    belly_center = np.array(meshes['Belly.obj'].centroid)

    targets = []
    for k in anatomy_list:
        bb_c = np.array(meshes[k].centroid)
        tgt = bb_c + (belly_center-bb_c)*0.6
        # (sanity) check if target is inside mesh 
        check_contains = []
        for j in anatomy_list:
            check_contains.extend(meshes[j].contains(np.expand_dims(tgt,0)))
            #print(k, j, np.min(np.array(meshes[j].vertices)-tgt))

        #assert not np.any(np.array(check_contains))
        targets.append(tgt)

    return meshes, vertices, targets

def check_inside_shell(state, c, r1, r2, theta_bounds=None, phi_bounds=None):
    if theta_bounds is None and phi_bounds is None:
        return np.linalg.norm(state-c) > r1 and np.linalg.norm(state-c) < r2
    else:
        state_ = state-c
        radius_condition = np.linalg.norm(state_) > r1 and np.linalg.norm(state_) < r2
        theta_ = np.arccos(state_[-1]/np.linalg.norm(state_))
        theta_condition = theta_ > theta_bounds[0] and theta_ < theta_bounds[1]
        #phi_ = np.sign(state_[1])*np.arccos(state_[0]/np.sqrt(state_[0]**2 + state_[1]**2))
        phi_ = np.arctan2(state_[1],state_[0])+np.pi/2
        phi_condition = phi_ > phi_bounds[0] and theta_ < phi_bounds[1]
                
        # print(radius_condition, theta_, theta_bounds, theta_condition, phi_, phi_bounds, phi_condition)
        return radius_condition and theta_condition and phi_condition

## plotting helpers
def plt_sphere(ax, list_center, list_radius, color, smoothness=50j):
    for c, r in zip(list_center, list_radius):

        # draw sphere
        u, v = np.mgrid[0:2*np.pi:smoothness, 0:np.pi:smoothness]
        x = r*np.cos(u)*np.sin(v)
        y = r*np.sin(u)*np.sin(v)
        z = r*np.cos(v)

        ax.plot_surface(x+c[0], y+c[1], z+c[2], color=color, alpha=0.1)
        
def plt_sphere_2(ax, list_center, list_radius, color, q = 0.5, p = 0.8):
    for c, r in zip(list_center, list_radius):     
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(q, p * np.pi, int(p*100))
        x = r * np.outer(np.cos(u), np.sin(v)) + c[0]
        y = r * np.outer(np.sin(u), np.sin(v)) + c[1]
        z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + c[2]
        ax.plot_surface(x, y, z, color=color, alpha=0.1)

def plt_sphere_section(ax, list_center, list_radius, color, u_range = (1.0,2.0),  v_range = (1.0,2.0)):
    for c, r in zip(list_center, list_radius):     
        #u = np.linspace(0, 2*np.pi, 100)
        #v = np.linspace(0, 2*np.pi, 100)
        
        # draw sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        
        u_ = u[(u > u_range[0]) & (u < u_range[1])]
        v_ = v[(v > v_range[0]) & (v < v_range[1])]
        
        x = r * np.outer(np.cos(u_), np.sin(v_)) + c[0]
        y = r * np.outer(np.sin(u_), np.sin(v_)) + c[1]
        z = r * np.outer(np.ones(np.size(u_)), np.cos(v_)) + c[2]
        ax.plot_surface(x, y, z, color=color, alpha=0.1)
        
class LapLiver(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, step_size=0.3, mesh_folder='demos/davos_data/obj', 
                 state_spec='r3', gc=False, condition_on_n_tasks_completed=True, 
                 condition_on_anatomy_distance=False, random_init=False,
                 done_on_collision=True, continuous_actions=True,
                 plot_meshes=True, collision_loss=True,
                 hard_collisions=False, ref_demo=None, 
                 vis_threshold_factor=100, render_mode='rgb_array'):
        super(LapLiver, self).__init__()

        self.render_mode = render_mode
        self.agent_render_q = deque(maxlen=100)

        self.name = 'lapliver'
        self.anatomy_list = ['LiverRight.obj', 'Falciform.obj', 'LiverLeft.obj']
        self.mesh_list = [
            'DomeOfBladder.obj', 'Uterus.obj', 'CecumAppendix.obj',
            'ColonRight.obj', 'LiverRight.obj', 'Falciform.obj',
            'LiverLeft.obj', 'GallBladder.obj', 'HepatoDuodenalLigament.obj',
            'Spleen.obj', 'Stomach.obj', 'Belly.obj']

        self.ref_demo = ref_demo

        # load meshes        
        self.meshes, self.vertices, self.waypoints = load_meshes_tri(mesh_folder, self.mesh_list)

        self.highlight_index_dict = {}
        anatomy_highlight_idx = {'LiverRight.obj': 1500, 'LiverLeft.obj':1072, 'Falciform.obj': 700}
        # this defines the valid occupancy spaces to visualize the necessary vertices
        self.occ_shell_specs = {'LiverRight.obj': [(0.0,np.pi), (np.pi/4,3*np.pi/2), 3,12], 
                                        'LiverLeft.obj': [(0.0,np.pi), (np.pi/3,3*np.pi/2), 8,20], 
                                        'Falciform.obj': [(0.0,np.pi), (np.pi/2,3*np.pi/2), 3,14]}

        self.target_balls = {'LiverRight.obj': [ 5.991351,  24.504538,  -1.9299495], 
                            'Falciform.obj': [ 5.056621,  22.632568,  -2.5371423],
                            'LiverLeft.obj': [ 4.9135485, 23.361105,   1.7277693]}

        self.curr_anatomy_reward = 0
        self.anatomy_vis_threshold = 500
        self.vis_threshold_factor = vis_threshold_factor

        # defines the highlighted vertex indices
        for anatomy_key in ['LiverRight.obj', 'LiverLeft.obj', 'Falciform.obj']:
            anatomy = self.meshes[anatomy_key].vertices
            mid_idx = anatomy_highlight_idx[anatomy_key]
            idx = np.arange(mid_idx,mid_idx+20)
            X = anatomy[idx]
            nbrs = NearestNeighbors(n_neighbors=30, algorithm='ball_tree').fit(anatomy)
            distances, indices = nbrs.kneighbors(X)
            self.highlight_index_dict[anatomy_key] = indices.reshape(-1)

        self.n_tgt = len(self.anatomy_list)
        #self.initial_position = np.array(self.meshes['DomeOfBladder.obj'].get_center()) + np.array([0.05,0.05,0.05]) 
        #self.initial_position = np.array(self.meshes['Belly.obj'].centroid) + np.array([0.0, 10.0, 0.0])

        # compute maximal distance between Belly centroid and 
        # vertex point to construct additional constraint ball
        dists = []
        self.belly_centroid = self.meshes['Belly.obj'].centroid

        for belly_vtx in self.meshes['Belly.obj'].vertices:
            dists.append(np.linalg.norm(self.belly_centroid-belly_vtx))

        self.max_belly_dist = np.max(dists)-10

        self.initial_position = np.array((10,17,0)) # XXX: check offsets

        # set movement constraints
        self.mesh_collision_radius = step_size
        belly_bbox = np.array(self.meshes['Belly.obj'].bounding_box_oriented.vertices)
        self.space_min = np.min(belly_bbox)
        self.space_max = np.max(belly_bbox)

        # setup flags
        self.goal_conditioning = gc
        self.condition_on_n_tasks_completed = condition_on_n_tasks_completed
        self.plot_meshes = plot_meshes
        self.done_on_collision = done_on_collision
        self.collision_loss = collision_loss
        self.hard_collisions = hard_collisions
        self.condition_on_anatomy_distance = condition_on_anatomy_distance

        self.render_cnt = 0
        # Define a 3-D or 7-D observation space

        if state_spec == 'r3so3':
            obs_dim = 7
        else:
            obs_dim = 3

        if self.goal_conditioning:
            obs_dim = obs_dim*2

        if condition_on_n_tasks_completed:
            obs_dim += 1

        if condition_on_anatomy_distance:
            obs_dim += len(self.meshes)

        self.observation_shape = (obs_dim,)
        self.observation_space = spaces.Box(
            low=np.ones(self.observation_shape,
                        dtype=np.float32)*self.space_min,
            high=np.ones(self.observation_shape,
                         dtype=np.float32)*self.space_max,
            dtype=np.float32)
    
        if state_spec == 'r3so3':
            self.action_space = spaces.Discrete(11,)
        else:
            self.action_space = spaces.Discrete(6,)

        self.continuous_actions = continuous_actions
        if self.continuous_actions:
            self.action_space = spaces.Box(
                low=np.ones([3])*-1, high=np.ones([3])*1, dtype=np.float32)
        
        # Define target points based on geometry
        
        # Maximum episode length
        self.max_time = 1000

        self.step_size = step_size
        self.random_init = random_init

        self.agent = self.initial_position
        self.terminated = False
        self.truncated = False
        self.time = self.max_time
        self.curr_tgt = 0
        self.needs_reset = False
        self.num_tasks_completed = []

        self.collisions = []
        self.ep_collisions = []
        
    def compute_endoscope_movement(state, action, portal_pos):
        # rotate 
        #this.transform.Rotate(this.transform.right, rotation_output * Time.deltaTime, Space.World);
        # Apply actions to transform: translation along forward direction
        #this.transform.Translate(this.transform.right * translation_output * Time.deltaTime, Space.World);

        #Apply actions to transform: pivot around portal
        #.transform.RotateAround(portal.transform.position, artcam.transform.up, pivot_output_x * Time.deltaTime);
        #.transform.RotateAround(portal.transform.position, -artcam.transform.right, pivot_output_y * Time.deltaTime);
        # rotate camera
        return state

    def reset(self, seed: int = None, **kwargs):
        self.terminated = False
        self.truncated = False
        self.time = self.max_time
        self.num_tasks_completed.append(self.curr_tgt)
        self.curr_tgt = 0
        self.curr_anatomy_reward = 0

        self.agent_render_q.clear()

        # Determine a place to intialise the agent in
        self.agent = self.initial_position
        if self.random_init:
            self.agent + np.random.randn(self.initial_position.shape)*0.1
        
        curr_tgt = np.array(self.waypoints[self.curr_tgt])
        if self.goal_conditioning:
            state = np.concatenate([self.agent, curr_tgt], 0)
        else:
            state = self.agent
        
        if self.condition_on_n_tasks_completed:
            state = np.concatenate([self.agent, curr_tgt, [self.curr_tgt]], -1)        

        if self.condition_on_anatomy_distance:
            # compute proximity
            dists = []
            for k in self.anatomy_list:
                mesh = self.meshes[k]
                _, dist, _ = trimesh.proximity.closest_point(mesh, [self.agent])
                dists.append(dist)
            dists = np.squeeze(np.array(dists))
            state = np.concatenate([state, dists], -1)
        self.needs_reset = False
        self.collisions.append(self.ep_collisions)
        self.ep_collisions = []

        state = state.astype(np.float32)
        info = {'terminal_observation': state}
        # return the observation
        return state, info

    def get_reward(self, state, tgt=(175, 175)):
        curr_tgt = np.array(tgt)
        # l2 distance as reward
        reward = -(np.sqrt((state[0] - curr_tgt[0])**2 + (state[1]-curr_tgt[1])**2))/1000
        return reward

    def compute_shell_occupancy_reward(self, state):
        curr_tgt_anatomy = self.anatomy_list[self.curr_tgt]
        oss = self.occ_shell_specs[curr_tgt_anatomy]
        h_idx = self.highlight_index_dict[curr_tgt_anatomy]
        v = self.meshes[curr_tgt_anatomy].vertices #XXX
        u_range = oss[0]
        v_range = oss[1]
        r1 = oss[2]
        r2 = oss[3]

        anatomy_reward = 0
        for idx in h_idx:
            c = v[idx,:]
            if check_inside_shell(state, c, r1,r2, u_range, v_range):
                anatomy_reward += 1

        return anatomy_reward, len(h_idx)


    def compute_ball_occupancy_reward(self, state, radius):
        curr_tgt_anatomy = self.anatomy_list[self.curr_tgt]
        ball_centroid = self.target_balls[curr_tgt_anatomy]

        if np.linalg.norm(state - ball_centroid) < radius:
            return 1
        else:
            return 0

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"
        self.time -= 1 

        # apply the action to the agent
        magnitude = self.step_size
        if self.continuous_actions:
            move = action
        else:
            if action == 0:
                move = (magnitude, 0, 0)
            elif action == 1:
                move = (-magnitude, 0, 0)
            elif action == 2:
                move = (0, magnitude, 0)
            elif action == 3:
                move = (0, -magnitude, 0)
            elif action == 4:
                move = (0, 0, magnitude)
            elif action == 5:
                move = (0, 0, -magnitude)
            else:
                move = (0, 0, 0)
        move = np.array(move)
        tentative_move = np.clip(
            self.agent + move, self.space_min, self.space_max)

        # Check for collisions
        reward = 0
        collided = False
        for k, v in self.meshes.items():
            dist = np.sum((tentative_move-v.vertices)**2, -1) < self.mesh_collision_radius
            if np.any(dist):
                collided = True
                self.ep_collisions.append(k)
                print(f"Collided with {k}")

                if self.collision_loss:
                    reward += -10000
                if self.done_on_collision:
                    self.terminated = True
                    self.needs_reset = True

        if np.linalg.norm(self.agent-self.belly_centroid) > self.max_belly_dist:
            print("Agent outside of Belly ball")
            reward += -10000
            self.terminated = True
            self.needs_reset = True

        if collided and self.hard_collisions:
            pass
        else:
            self.agent = tentative_move

        curr_tgt = np.array(self.waypoints[self.curr_tgt])
        # l2 distance as reward
        #reward += -np.linalg.norm(self.agent - curr_tgt)/1000

        #anatomy_reward, vis_thresh = self.compute_shell_occupancy_reward(self.agent)
        anatomy_reward = self.compute_ball_occupancy_reward(self.agent, radius=3.5)

        self.curr_anatomy_reward += anatomy_reward
        reward += anatomy_reward

        #print(self.curr_anatomy_reward, vis_thresh*100)
        #if self.curr_anatomy_reward >= self.anatomy_vis_threshold:
        #if self.curr_anatomy_reward >= vis_thresh*self.vis_threshold_factor:
        print(self.curr_anatomy_reward)
        if self.curr_anatomy_reward >= self.vis_threshold_factor:
            print(f"Task solved: {self.anatomy_list[self.curr_tgt]} reached ")

            if self.curr_tgt == len(self.anatomy_list)-1:
                # task solved
                reward += 1000
                #self.curr_tgt += 1
                self.curr_anatomy_reward = 0
                self.terminated = True
            else:
                # update target
                reward += 500
                # self.time += 500
                self.curr_tgt += 1
                self.curr_anatomy_reward = 0

        if self.goal_conditioning:
            state = np.concatenate([self.agent, curr_tgt], -1)
        else:
            state = self.agent
        
        if self.condition_on_n_tasks_completed:
            state = np.concatenate([self.agent, curr_tgt, [self.curr_tgt]], -1)

        if self.condition_on_anatomy_distance:
            # compute proximity
            dists = []
            for k in self.anatomy_list:
                mesh = self.meshes[k]
                _, dist, _ = trimesh.proximity.closest_point(
                    mesh, [self.agent])
                dists.append(dist)
            dists = np.squeeze(np.array(dists))
            state = np.concatenate([state, dists], -1)

        # If out of fuel, end the episode.
        if self.time == 0:
            self.truncated = True
            self.needs_reset = True

        state = state.astype(np.float32)
        info = {'terminal_observation': state}
        return state, reward, self.terminated, self.truncated, info

    def render(self, mode="rgb_array"):
        # do 3d plot with meshes
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        #matplotlib.style.use('seaborn')

        #fig = plt.gcf()
         
        fig = plt.figure(figsize=[12, 4], dpi=200)

        views = ((-50,90), (0,90), (25,36))

        for i, view in enumerate(views):
            ax = fig.add_subplot(1,3,i+1, projection='3d')
            #ax = Axes3D(fig)
            # TODO: parametrize viewpoint to follow X? Alternatively, 3 perspectives?
            ax.view_init(*view)
            ax.set_box_aspect((1, 1, 1), zoom=2.0)

            scale = 1
            self.render_cnt += 1
            #print("Render count: ", self.render_cnt, flush=True)
            #ax.view_init(-10, 90)
            total_points = 0
            for k, v in self.meshes.items():
                total_points += len(v.vertices)
                vtx = np.array(v.vertices)
                #bb = np.array(v.get_oriented_bounding_box().get_box_points())
                #bb_c = np.array(v.get_oriented_bounding_box().get_center())
                bb = np.array(v.bounding_box_oriented.vertices)*scale
                bb_c = np.array(v.centroid)*scale
                y_offset = 9.8
                if self.plot_meshes:
                    if k in self.anatomy_list:
                        idx = self.highlight_index_dict[k]
                        ax.scatter(vtx[:, 0], vtx[:, 1]+y_offset, vtx[:, 2], s=0.2)
                        ax.scatter(vtx[idx, 0], vtx[idx, 1]+y_offset, vtx[idx, 2],
                                s=3, c='red')
                    else:
                        ax.scatter(vtx[:, 0], vtx[:, 1]+y_offset, vtx[:, 2], s=0.1, c='gray')
                        #ax.scatter(vtx[idx, 0], vtx[idx, 1], vtx[idx, 2],
                        #           s=3, c='cyan')

                    #if 'Belly' in k:
                    #    plt_sphere(ax, list(np.array([vtx[:, 0], vtx[:, 1]+y_offset, vtx[:, 2]]).T), 
                    #                list(np.ones_like(vtx[:,0])*self.step_size), color='yellow')


                #ax.scatter(bb[:,0],bb[:,1],bb[:,2], s=10, c='red')
                ax.scatter(bb_c[0], bb_c[1], bb_c[2], s=10, c='yellow')

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

            if self.ref_demo is not None:
                v = self.ref_demo['all']['obs']
                v_f = self.ref_demo['all']['obs_full']

                liver_states = v[np.where(v_f[:,-1]==12+self.curr_tgt),:].squeeze()
                ax.scatter(liver_states[:,0],liver_states[:,1],-liver_states[:,2], s=3, c='blue')
                ax.plot(liver_states[:,0],liver_states[:,1],-liver_states[:,2])

            # plot target areas:
            curr_anat = self.anatomy_list[self.curr_tgt]
            u_range = self.occ_shell_specs[curr_anat][0]
            v_range = self.occ_shell_specs[curr_anat][1]
            r1 = self.occ_shell_specs[curr_anat][2]
            r2 = self.occ_shell_specs[curr_anat][3]
            idx = self.highlight_index_dict[curr_anat][0]
            v = np.array(self.meshes[curr_anat].vertices)

            #plt_sphere_section(ax, [(v[idx,0],v[idx,1]+9.8,v[idx,2])], [r1], color='yellow', u_range = u_range,  v_range = v_range)
            #plt_sphere_section(ax, [(v[idx,0],v[idx,1]+9.8,v[idx,2])], [r2], color='yellow', u_range = u_range,  v_range = v_range)

            plt_sphere(ax, [self.target_balls[curr_anat]], [3.5], color='magenta')

            # plot agent
            self.agent_render_q.append(self.agent)
            for i, agent in enumerate(self.agent_render_q):
                ax.plot(agent[0], agent[1], agent[2], c='red')
                ax.scatter(agent[0], agent[1], agent[2], s=20, c='red', alpha=(i+1)/(len(self.agent_render_q)+1))

            wp = np.array(self.waypoints)
            ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2], s=30, c='green')
            wp_curr = wp[self.curr_tgt]

            ax.scatter(wp_curr[0], wp_curr[1], wp_curr[2], s=40, c='m')
            
            # disable grid and axes
            plt.axis('off')
            plt.grid(b=None)

        # plot demos?
        if mode == "human":
            plt.show()
            plt.close()


        elif mode == "rgb_array":
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            plt.close()
            return data[:,:,:3]


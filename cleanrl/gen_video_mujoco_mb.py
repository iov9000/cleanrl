import os
import mujoco
import mujoco.viewer
import mujoco_viewer # alternative viewer 

import numpy as np
import argparse
from datetime import datetime

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image
from mpl_toolkits.mplot3d import Axes3D

import xml.etree.ElementTree as ET


def create_custom_humanoid(positions, inertiafromgeom=False, floor_z_offset=0.0):
    def create_torso(parent, name, pos_torso, pos_head, pos_neck, 
                     pos_right_shoulder, pos_left_shoulder, rgba=[0.8, 0.6, 0.6, 1]):
        body = ET.SubElement(parent, 'body', name=name, pos=" ".join(map(str, pos_torso)))
        ET.SubElement(body, 'camera', mode="track", pos="0 0 0", xyaxes="1 0 0 0 0 1")
        #fromto = "0 -.07 0 0 .07 0"
        fromto = " ".join(map(str, pos_right_shoulder)) + " " + " ".join(map(str, pos_left_shoulder))

        ET.SubElement(body, 'geom', type="capsule", fromto=fromto, size=".05", rgba=" ".join(map(str, rgba)))
        #ET.SubElement(body, 'geom', type="sphere", size="0.07", rgba=" ".join(map(str, rgba)))

        ET.SubElement(body, 'geom', name="head", pos=" ".join(map(str, pos_head)), size=".08", 
                      type="sphere", rgba=" ".join(map(str, rgba)))
        #ET.SubElement(body, 'geom', name="neck", pos=" ".join(map(str, pos_neck)), size=".04", 
        #              type="sphere", rgba=" ".join(map(str, rgba)))
        ET.SubElement(body, 'joint', name="root", pos="0 0 0", 
                      type="free", armature="0", damping="0", stiffness="0", limited="false")

        return body
        
    def create_abdomen(parent, name, pos_abdomen, pos_pelvis, rgba=[0.8, 0.6, 0.6, 1]):
        body = ET.SubElement(parent, 'body', name=name, pos=" ".join(map(str, pos_abdomen)),
                            quat="1.000 0 -.002 0")
        ET.SubElement(body, 'geom', pos="0 0 0.1", name="abdomen", size=".08", type="sphere")
        fromto = " ".join(map(str,[0, 0, 0])) + " " +  " ".join(map(str,pos_pelvis))
        ET.SubElement(body, 'geom', type="capsule", fromto=fromto, size=".05", rgba=" ".join(map(str, rgba)))


        ET.SubElement(body, 'joint', name="abdomen_z", pos="0 0 0.065", axis="0 0 1",
                      type="hinge", armature="0.02", range="-45 45", damping="5", stiffness="20")
        ET.SubElement(body, 'joint', name="abdomen_y", pos="0 0 0.065", axis="0 1 0",
                      type="hinge", armature="0.02", range="-75 30", damping="5", stiffness="10")

        return body
        
    def create_pelvis(parent, name, pos_pelvis, pos_right_hip, pos_left_hip, rgba=[0.8, 0.6, 0.6, 1]):
        body = ET.SubElement(parent, 'body', name=name, pos=" ".join(map(str, pos_pelvis)))
        
        #from_ = pos_pelvis + np.array([-.01,-.07,0])
        #to = pos_pelvis + np.array([-.01,.07,0])
        fromto = " ".join(map(str, pos_right_hip)) + " " + " ".join(map(str, pos_left_hip))
        #fromto = "-.02 -.07 0 -.02 .07 0"
        ET.SubElement(body, 'geom', type="capsule", size="0.05", 
                      fromto=fromto, rgba=" ".join(map(str, rgba)))
        
        #ET.SubElement(body, 'geom', type="sphere", size="0.07", rgba=" ".join(map(str, rgba)))
        ET.SubElement(body, 'joint', name="abdomen_x", pos="0 0 0.1", axis="1 0 0",
                      type="hinge", armature="0.02", range="-35 35", damping="5", stiffness="10")    

        return body
        
    def create_leg(parent, name, pos_thigh, pos_shin, pos_foot,
                   thigh_capsule_fromto=[0, -0.1, 0, 0, 0.01, -.34],
                   shin_capsule_fromto=[0, 0, 0, 0, 0, -.3], rgba=[0.8, 0.6, 0.6, 1], side="left"):
        body = ET.SubElement(parent, 'body', name=name, pos=" ".join(map(str, pos_thigh)))
        
        d = {"left": -1, "right": 1}
        
        ET.SubElement(body, 'joint', name=side + "_hip_x", pos="0 0 0", axis=str(d[side]) + " 0 0",
                      type="hinge", armature="0.01", range="-25 5", damping="5", stiffness="10") 
        ET.SubElement(body, 'joint', name=side + "_hip_z", pos="0 0 0", axis="0 0 " + str(d[side]),
                      type="hinge", armature="0.01", range="-60 35", damping="5", stiffness="10") 
        ET.SubElement(body, 'joint', name=side + "_hip_y", pos="0 0 0", axis="0 1 0",
                      type="hinge", armature="0.01", range="-110 20", damping="5", stiffness="20") 
    
        fromto_thigh = " ".join(map(str,[0, 0, 0])) + " " +  " ".join(map(str,pos_shin))
        
        #fromto_thigh = thigh_capsule_fromto
        #fromto_thigh[4] = fromto_thigh[4]*d[side]
        #fromto_thigh = str(fromto_thigh).replace(',','').replace('[','').replace(']','')
        
        ET.SubElement(body, 'geom', type="capsule", name=side + "_thigh",
                      fromto=fromto_thigh, size="0.06", rgba=" ".join(map(str, rgba)))
        

        shin = ET.SubElement(body, 'body', name=side+"_shin", pos=" ".join(map(str, pos_shin)))
        fromto_shin = " ".join(map(str,[0, 0, 0])) + " " +  " ".join(map(str,pos_foot))
        #fromto_shin = str(shin_capsule_fromto).replace(',','').replace('[','').replace(']','')
        ET.SubElement(shin, 'joint', name=side + "_knee", pos="0 0 0", axis=str(d[side]) + " 0 0",
                      type="hinge", armature="0.006", range="-160 -2", damping="5", stiffness="10") 
        ET.SubElement(shin, 'geom', fromto=fromto_shin, name=side+"_shin1", size="0.049", type="capsule")
        
        foot = ET.SubElement(shin, 'body', name=side+"_foot", pos=" ".join(map(str, pos_foot)))
        ET.SubElement(foot, 'geom', type="sphere", name=side + "_foot", pos="0 0 0",
                      size="0.07", rgba=" ".join(map(str, rgba)))
        
        return body, shin, foot

        
    def create_arm(parent, name, pos_shoulder, pos_elbow, pos_hand, rgba=[0.8, 0.6, 0.6, 1], side="left"):
        body = ET.SubElement(parent, 'body', name=name, pos=" ".join(map(str, pos_shoulder)))
        
        d = {"left": -1, "right": 1}

        ET.SubElement(body, 'joint', name=side + "_shoulder1", pos="0 0 0", axis="2 " + str(d[side]) + " 1",
                      type="hinge", armature="0.0068", range="-85 60", stiffness="1") 
        ET.SubElement(body, 'joint', name=side + "_shoulder2", pos="0 0 0", axis="0 " + str(d[side]*-1) + " 1",
                      type="hinge", armature="0.0051", range="-85 60", stiffness="1") 
    
        fromto_upper = " ".join(map(str, [0, 0, 0])) + " " +  " ".join(map(str, pos_elbow))
        #fromto_upper = "0.0 0.0 0.0 .16 " + str(d[side]*-.16) + " -.16"

        ET.SubElement(body, 'geom', type="capsule", name=side + "_upper_arm", fromto=fromto_upper,
                      size="0.04 0.16", rgba=" ".join(map(str, rgba)))     
        
        lower_arm = ET.SubElement(body, 'body', name=side+"_lower_arm", pos=" ".join(map(str, pos_elbow)))
        fromto_lower = " ".join(map(str, [0, 0, 0])) + " " +  " ".join(map(str, pos_hand))
        #fromto_lower = "0.01 0.01 0.01 .17 " + str(d[side]*.17) + " -.16"
        ET.SubElement(lower_arm, 'geom', type="capsule", name=side + "_larm", fromto=fromto_lower,
                      size="0.031", rgba=" ".join(map(str, rgba)))     
        ET.SubElement(lower_arm, 'joint', name=side + "_elbow", pos="0 0 0", axis="0 -1 " + str(d[side]),
                      type="hinge", armature="0.0028", range="-90 50", stiffness="1")         
        ET.SubElement(lower_arm, 'geom', type="sphere", name=side + "_hand", pos=" ".join(map(str, pos_hand)),
                      size="0.04", rgba=" ".join(map(str, rgba)))     
        
        return body, lower_arm
        
    def create_tendons(parent):
        tendon = ET.SubElement(parent, 'tendon')
        left_hipknee = ET.SubElement(tendon, 'fixed', name="left_hipknee")
        ET.SubElement(left_hipknee, 'joint', coef="-1", joint="left_hip_y")
        ET.SubElement(left_hipknee, 'joint', coef="1", joint="left_knee")
        right_hipknee = ET.SubElement(tendon, 'fixed', name="right_hipknee")
        ET.SubElement(right_hipknee, 'joint', coef="-1", joint="right_hip_y")
        ET.SubElement(right_hipknee, 'joint', coef="1", joint="right_knee")

        
    # take absolute positions from dict and compute relative positions according to limb hierarchy
    pos_torso = positions["torso"]
    pos_head = positions["head"] - pos_torso
    pos_neck = positions["neck"] - pos_torso
    
    pos_abdomen = positions["abdomen"] - pos_torso
    pos_pelvis = positions["pelvis"] - positions["abdomen"]
    
    pos_right_thigh = positions["right_hip"] - positions["pelvis"]
    pos_right_knee = positions["right_knee"] - positions["right_hip"]
    pos_right_foot = positions["right_foot"] - positions["right_knee"]
    
    pos_left_thigh = positions["left_hip"] - positions["pelvis"]
    pos_left_knee = positions["left_knee"] - positions["left_hip"]
    pos_left_foot = positions["left_foot"] - positions["left_knee"]
    
    pos_left_shoulder = positions["left_shoulder"] - pos_torso
    pos_left_elbow = positions["left_elbow"] - positions["left_shoulder"]
    pos_left_hand = positions["left_hand"] - positions["left_elbow"]

    pos_right_shoulder = positions["right_shoulder"] - pos_torso
    pos_right_elbow = positions["right_elbow"] - positions["right_shoulder"]
    pos_right_hand = positions["right_hand"] - positions["right_elbow"]
    
    #print(pos_torso, pos_head, pos_neck, pos_abdomen, pos_pelvis, pos_right_thigh, pos_right_foot, pos_left_shoulder)

    ## preamble
    # Create the root element
    model = ET.Element('mujoco', model="custom_humanoid")
    ET.SubElement(model, 'compiler', angle="degree", inertiafromgeom=str(inertiafromgeom).lower())
    default_preamble = ET.SubElement(model, 'default')
    ET.SubElement(default_preamble, 'joint', armature="1", damping="1", limited="true")
    ET.SubElement(default_preamble, 'geom', conaffinity="1", condim="1", contype="1", margin="0.001",
                material="geom", rgba="0.8 0.6 0.4 1")
    ET.SubElement(default_preamble, 'motor', ctrllimited="true", ctrlrange="-.4 .4")
    ET.SubElement(model, 'option', integrator="RK4", solver="PGS", timestep="0.003")
    ET.SubElement(model, 'size', nkey="5", nuser_geom="1")

    plane = ET.SubElement(model, 'asset')
    ET.SubElement(plane, 'texture', builtin="gradient", height="100", 
                rgb1=".4 .5 .6", rgb2="0 0 0", type="skybox", width="100")
    ET.SubElement(plane, 'texture', builtin="flat", mark="cross", height="1278",
                markrgb="1 1 1", name="texgeom", random="0.01", 
                rgb1="0.8 0.6 0.4", rgb2="0.8 0.6 0.4", type="cube", width="127")
    ET.SubElement(plane, 'texture', builtin="checker", height="100", name="texplane",
                rgb1="0 0 0", rgb2="0.8 0.8 0.8", type="2d", width="100")           
    ET.SubElement(plane, 'material', name="MatPlane", reflectance="0.5", 
                shininess="1", specular="1", texrepeat="60 60", texture="texplane")   
    ET.SubElement(plane, 'material', name="geom", texture="texgeom", texuniform="true")
    ### create the parts 

    # Create the worldbody element
    worldbody = ET.SubElement(model, 'worldbody')
    ET.SubElement(worldbody, 'light', cutoff="100", diffuse="1 1 1", 
        dir="-0 0 -1.3", directional="true", exponent="1", 
        pos="0 0 1.3" , specular=".1 .1 .1")
    ET.SubElement(worldbody, 'geom', condim="3",
        friction="1 .1 .1", material="MatPlane", name="floor",
        pos="0 0 "+str(floor_z_offset), rgba="0.8 0.9 0.8 1", size="20 20 0.125", type="plane")

    torso = create_torso(worldbody, "torso", pos_torso, pos_head, pos_neck, pos_right_shoulder, pos_left_shoulder)

    abdomen = create_abdomen(torso, "abdomen", pos_abdomen, pos_pelvis)
    pelvis = create_pelvis(abdomen, "pelvis", pos_pelvis, pos_right_thigh, pos_left_thigh)
    lleg = create_leg(pelvis, "left_leg", pos_left_thigh, pos_left_knee, pos_left_foot, side="left")
    rleg = create_leg(pelvis, "right_leg", pos_right_thigh, pos_right_knee, pos_right_foot, side="right")
    larm = create_arm(torso, "left_arm", pos_left_shoulder, pos_left_elbow, pos_left_hand, side="left")
    rarm = create_arm(torso, "right_arm", pos_right_shoulder, pos_right_elbow, pos_right_hand, side="right")
    create_tendons(model)
    
    # Create actuators for the joints
    actuated_joints = {"abdomen_x": 100, "abdomen_y": 100, "abdomen_z":100,
                      "right_hip_x":100, "right_hip_z":100, "right_hip_y":300,
                      "right_knee":200, "left_hip_x":100, "left_hip_z":100,
                      "left_hip_y":300, "left_knee":200, "right_shoulder1":25,
                      "right_shoulder2":25, "right_elbow":25, "left_shoulder1":25,
                      "left_shoulder2":25, "left_elbow":25}
    
    
    actuator = ET.SubElement(model, 'actuator')
    for joint,gear in actuated_joints.items():
        ET.SubElement(actuator, 'motor', gear=str(gear), joint=joint, name=joint)#, ctrlrange="-1 1")
       
    # Convert the ElementTree to a string
    tree = ET.ElementTree(model)
    ET.indent(tree, space="\t", level=0)
    from io import StringIO
    output = StringIO()
    tree.write(output, encoding='unicode', xml_declaration=True)
    return output.getvalue()

def plot_3d_connections(data, elev=12., azim=80, motionbert_coordinates=True):
    """
    Generates a 3D plot from a dictionary of names and positions with lines connecting the points.
    
    Parameters:
    - data (dict): A dictionary where keys are names and values are tuples representing (x, y, z) coordinates.
    """
    
    # Extract names and positions
    names = list(data.keys())
    positions = np.array(list(data.values()))
    
    # Create a new figure and 3D axis
    fig = plt.figure(figsize=(4,4), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    # Plot points
    #ax.scatter(-positions[:, 0], -positions[:, 2], -positions[:, 1], color='b', s=40, label='Points')
    
    # Annotate points with names
    for name, (x, y, z) in zip(names, positions):
        if motionbert_coordinates:
            ax.text(-x, -z, -y, name, fontsize=8, ha='right')
        else:
            ax.text(x, y, z, name, fontsize=8, ha='right')

    
    # Plot lines connecting points
    #print(data["torso"])

    limbSeq = [["pelvis", "right_hip"], ["right_hip", "right_knee"], ["right_knee", "right_foot"], 
               ["pelvis", "left_hip"], ["left_hip", "left_knee"], ["left_knee", "left_foot"], 
               ["pelvis", "abdomen"], ["abdomen", "torso"], ["torso", "neck"], ["neck", "head"],  
               ["torso", "left_shoulder"], ["left_shoulder", "left_elbow"], ["left_elbow", "left_hand"], 
               ["torso", "right_shoulder"], ["right_shoulder", "right_elbow"], ["right_elbow", "right_hand"]]

    for limb in limbSeq:
        i = limb[0]
        j = limb[1]
        
        x_values = np.array([data[i][0], data[j][0]])
        y_values = np.array([data[i][1], data[j][1]])
        z_values = np.array([data[i][2], data[j][2]])
        
        if motionbert_coordinates:
            ax.plot(-x_values, -z_values, -y_values, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2)
        else:
            ax.plot(x_values, y_values, z_values, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2)

    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    # Show plot
    plt.legend()
    plt.show()

#%matplotlib inline
#plot_3d_connections(get_index(position_seq_dict,1450))

def convert_coord_motionbert_to_mujoco(positions_seq_dict, gain=1.0):
    d = {}
    
    min_z = -100
    for k,v in position_seq_dict.items():
        y_mb = -v[:,1]*gain
        min_z_limb = np.min(y_mb)
        if min_z_limb < min_z:
            min_z = min_z_limb

    for k,v in position_seq_dict.items():
        x_mb = v[:,0]*gain 
        z_mb = v[:,2]*gain
        
        # add global offset to z to avoid collisions with floor
        y_mb = -v[:,1]*gain
        y_mb += np.abs(min_z_limb) + 1
        d[k] = np.stack([-x_mb, -z_mb, y_mb],-1)

    return d

def get_index(pos_seq_dict, index):
    d = {}
    for k,v in pos_seq_dict.items():
        d[k] = v[index, :]

    return d


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--mb_data_file", type=str, default="X3D.py", help="path to data extracted via MotionBert")
    
    args = parser.parse_args()
    # fmt: on
    non_default = {
        opt.dest: getattr(args, opt.dest)
        for opt in parser._option_string_actions.values()
        if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
    }

    return args, non_default

if __name__ == "__main__":
    args, non_default = parse_args()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    
    poses = np.load("X3D.npy")
    print(poses.shape)

    position_seq_dict = {"pelvis": poses[:,0,:],
                         "right_hip": poses[:,1,:],
                         "right_knee": poses[:,2,:],
                         "right_foot": poses[:,3,:],
                         "left_hip": poses[:,4,:],
                         "left_knee": poses[:,5,:],
                         "left_foot": poses[:,6,:],
                         "abdomen": poses[:,7,:],
                         "torso": poses[:,8,:],
                         "neck": poses[:,9,:], # swapped head and neck
                         "head": poses[:,10,:],
                         "left_shoulder": poses[:,11,:],
                         "left_elbow": poses[:,12,:],
                         "left_hand": poses[:,13,:],
                         "right_shoulder": poses[:,14,:],
                         "right_elbow": poses[:,15,:],
                         "right_hand": poses[:,16,:]}

    # TODO: loop over position_seq
    frames = []
    for i in tqdm(range(len(poses))):
    #for i in tqdm(range(100)):

        tf_dict = get_index(convert_coord_motionbert_to_mujoco(position_seq_dict,gain=3.0), i)
        mjcf_description = create_custom_humanoid(tf_dict, inertiafromgeom=True)

        model = mujoco.MjModel.from_xml_string(mjcf_description)
        #model = mujoco.MjModel.from_xml_path('humanoid.xml')

        data = mujoco.MjData(model)
    
        viewer = mujoco_viewer.MujocoViewer(model, data, 'offscreen', width=600, height=400)
        mujoco.mj_forward(model, data)

        viewer.cam.distance = 3
        viewer.cam.elevation=-10
        viewer.cam.lookat =np.array([ 0.0 , 0.0 , 1.5 ])
        #viewer.render()
        img = viewer.read_pixels(camid=-1)
        #frames.append(np.transpose(img, (0,2,1)))
        frames.append(img)
        viewer.close()
    
    import cv2
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1200, 800))
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        out.write(frame) # frame is a numpy.ndarray with shape (1280, 720, 3)
    out.release()
    #matplotlib.image.imsave('test_.png', img)

    # # create openGL context
    # ctx = mujoco.GLContext(400,400)
    # ctx.make_current()
    # con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    # # #mujoco.glfw.glfw.init()
    # # #window = mujoco.glfw.glfw.create_window(1200, 900, "Demo", None, None)
    # # #mujoco.glfw.glfw.make_context_current(window)
    # # #mujoco.glfw.glfw.swap_interval(1)

    # viewport = mujoco.MjrRect(0, 0, 400, 400)
    # scn = mujoco.MjvScene(model, maxgeom=1000)
    # cam = mujoco.MjvCamera()
    # cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    # #cam.fixedcamid = -1
    # #for i in range(3):
    # #    cam.lookat[i] = np.mean(data.geom_xpos[:, i])
    # #cam.distance = model.stat.extent
    # #cam.azimuth=90
    # vopt = mujoco.MjvOption()
    # pert = mujoco.MjvPerturb()
    
    # mujoco.mjv_defaultCamera(cam)
    # mujoco.mjv_defaultOption(vopt)

    # mujoco.mj_forward(model, data)

    # mujoco.mjv_updateScene(
    #         model,
    #         data,
    #         vopt,
    #         pert,
    #         cam,
    #         mujoco.mjtCatBit.mjCAT_ALL,
    #         scn,
    # )

    # mujoco.mjr_render(viewport, scn, con)

    # #mujoco.glfw.glfw.swap_buffers(window)
    # #mujoco.glfw.glfw.poll_events()

    # rgb_arr = np.zeros(
    #     3 * viewport.width * viewport.height, dtype=np.uint8
    # )
    # depth_arr = np.zeros(
    #     viewport.width * viewport.height, dtype=np.float32
    # )

    # mujoco.mjr_readPixels(rgb_arr, depth_arr, viewport, con)
    # rgb_img = rgb_arr.reshape(viewport.height, viewport.width, 3)

    # #plt.imshow(rgb_img[::-1, :, :])
    # matplotlib.image.imsave('test.png', rgb_img[::-1, :, :])

    # #viewer = mujoco.viewer.launch_passive(model, data)
    # #viewer = mujoco.viewer.launch(model, data)

    # #mujoco.glfw.glfw.terminate()

    # # context = mujoco.MjrContext(model, mujoco.mjcf_to_mjmodel(model))
    # #mujoco.mjr_render(800, 600, context, mujoco.mjv_make_framebuffer(model, data))

    # #print(dir(viewer.cam))
    # # offscreen rendering
    # #renderer = mujoco.Renderer(m)
    # #renderer.update_scene(d)
    # #rgb_arr = renderer.render()
    # #print(rgb_arr.shape)

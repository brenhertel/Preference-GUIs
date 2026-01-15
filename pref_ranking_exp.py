import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--user")

args = parser.parse_args()

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('qtagg')

import os
import sys
import json
import copy
from math import pi
import h5py
import time
from utils import *
from downsampling import *


import rospy
import roslib
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from myviz import *

from std_msgs.msg import Int32

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


#How the robot is understood and controlled
class MoveGroupPythonInterface(object):
    def __init__(self):
        super(MoveGroupPythonInterface, self).__init__()
        #the moveit_commander is what is responsible for sending info the moveit controllers
        print(sys.argv)
        moveit_commander.roscpp_initialize(sys.argv)
        #initialize node
        #rospy.init_node('demo_xyz_playback', anonymous=True)
        #Instantiate a `RobotCommander`_ object. Provides information such as the robot's kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()
        #Instantiate a `PlanningSceneInterface`_ object. This provides a remote interface for getting, setting, and updating the robot's internal understanding of the surrounding world:
        scene = moveit_commander.PlanningSceneInterface()
        #Instantiate a `MoveGroupCommander`_ object.  This object is an interface to a planning group (group of joints), which in our moveit setup is named 'manipulator'
        group_name = "manipulator"
        move_group = moveit_commander.MoveGroupCommander(group_name)
        #Create a `DisplayTrajectory`_ ROS publisher which is used to display trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)
        self.gripper_publisher = rospy.Publisher('/gripper_sends/position', Int32, queue_size=20)
     
        #Get all the info which is carried with the interface object
        #We can get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        #print "Planning frame: %s" % planning_frame

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        #print  End effector link: %s" % eef_link

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        #print  Available Planning Groups:", robot.get_group_names()

        # Misc variables
        self.box_name1 = ''
        self.box_name2 = ''
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        
        self.demo_name = None
        self.repro_traj = None
        
    def set_demo(self, demo_name):
        self.demo_name = demo_name
        
    def set_repro(self, repro):
        self.repro_traj = repro

    def goto_joint_state(self, js_array, i=0):
        # To start the playback of the demo, go to the initial demo position, which can be interpreted as the 0th set of joint states
        # I use joint states instead of cartesians because cartesians will fail if the current state is too far away from the goal state, whereas joint states will simply execute
        joint_goal = self.move_group.get_current_joint_values()
        print(joint_goal)
        joint_goal[0] = js_array[i][2]
        joint_goal[1] = js_array[i][1]
        joint_goal[2] = js_array[i][0]
        joint_goal[3] = js_array[i][3]
        joint_goal[4] = js_array[i][4]
        joint_goal[5] = js_array[i][5]
        # go to the initial position
        # The go command can be called with joint values, poses, or without any parameters if you have already set the pose or joint target for the group
        self.move_group.go(joint_goal, wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group.stop()
      
    def goto_xyz(self, pos_data, rot_data, i=0):
        wpose = self.move_group.get_current_pose().pose
        wpose.position.x = -pos_data[i][0]
        wpose.position.y = -pos_data[i][1]
        wpose.position.z = pos_data[i][2]
        wpose.orientation.x = -rot_data[i][1]
        wpose.orientation.y = rot_data[i][0]
        wpose.orientation.z = rot_data[i][3]
        wpose.orientation.w = -rot_data[i][2]
        waypoints = []
        waypoints.append(copy.deepcopy(wpose))
        (start_plan, start_fraction) = self.move_group.compute_cartesian_path(waypoints, 0.001, 0.0)
        self.move_group.execute(start_plan, wait=True)
      
    def goto_xyz2(self, pos_data, rot_data):
        wpose = self.move_group.get_current_pose().pose
        wpose.position.x = -pos_data[0]
        wpose.position.y = -pos_data[1]
        wpose.position.z = pos_data[2]
        wpose.orientation.x = -rot_data[1]
        wpose.orientation.y = rot_data[0]
        wpose.orientation.z = rot_data[3]
        wpose.orientation.w = -rot_data[2]
        waypoints = []
        waypoints.append(copy.deepcopy(wpose))
        (start_plan, start_fraction) = self.move_group.compute_cartesian_path(waypoints, 0.001, 0.0)
        self.move_group.execute(start_plan, wait=True)
        
    def execute_joint_path(self, scale=1):
        #start planning demo playback by reading data from the demo.h5 file
        
        #ask user for the file which the playback is for
        #filename = raw_input('Enter the filename of the .h5 demo: ')
        filename = self.demo_name
        #open the file
        hf = h5py.File(filename, 'r')
        #navigate to necessary data and store in numpy arrays
        tf_info = hf.get('transform_info')
        js_info = hf.get('joint_state_info')
        pos_data = tf_info.get('transform_positions')
        rot_data = tf_info.get('transform_orientations')
        pos_data = np.array(pos_data)
        rot_data = np.array(rot_data)
        js_data = js_info.get('joint_positions')
        js_data = np.array(js_data)
        #close out file
        hf.close()
        

        #### move to starting position ####
        self.goto_joint_state(js_data, 0)
        print("Press 'Enter' to start")
        input()
        
        for i in range(1, len(js_data)):
        	self.goto_joint_state(js_data, i)
     
        return 
        
    def exec_cartesian_path(self, scale=1):
        #start planning demo playback by reading data from the demo.h5 file
        
        #ask user for the file which the playback is for
        #filename = raw_input('Enter the filename of the .h5 demo: ')
        filename = self.demo_name
        #open the file
        hf = h5py.File(filename, 'r')
        #navigate to necessary data and store in numpy arrays
        tf_info = hf.get('transform_info')
        js_info = hf.get('joint_state_info')
        pos_data = tf_info.get('transform_positions')
        rot_data = tf_info.get('transform_orientations')
        pos_data = np.array(pos_data)
        rot_data = np.array(rot_data)
        js_data = js_info.get('joint_positions')
        js_data = np.array(js_data)
        #close out file
        hf.close()
        
        
        #### move to starting position ####
        self.goto_joint_state(js_data, 0)
        
        print('start')
        print(self.repro_traj[0, :])
        print('end')
        print(self.repro_traj[-1, :])
        (n_pts, n_dims) = np.shape(self.repro_traj)
        
        self.goto_xyz(repro_traj, rot_data)

        (n_pts_og, _) = np.shape(rot_data)
        og_moments = [0.0, 0.25, 0.5, 0.75, 1.0]
        moments = [0.0, 0.25, 0.5, 0.75, 1.0]
        true_inds = [int(moments[i] * (n_pts_og - 1)) for i in range(len(og_moments))]
        key_inds = [int(moments[i] * (n_pts - 1)) for i in range(len(moments))]
        key_rots = R.from_quat([ [rot_data[ind][0], rot_data[ind][1], rot_data[ind][2], rot_data[ind][3]] for ind in true_inds])
        #print(key_rots.as_quat())
        #print(true_inds)
        #print(key_inds)
        
        slerp = Slerp(key_inds, key_rots)

        print('Press enter to execute')
        input()
        print('Executing')
        waypoints = []
        wpose = self.move_group.get_current_pose().pose
        for i in range(1, n_pts):
          
            cur_R = slerp(np.array([i]))
            cur_quats = cur_R.as_quat()
            self.goto_xyz2(repro_traj[i, :], cur_quats[0])
        
        return
        
    def plan_cartesian_path(self, scale=1):
        #start planning demo playback by reading data from the demo.h5 file
        
        #ask user for the file which the playback is for
        #filename = raw_input('Enter the filename of the .h5 demo: ')
        print(self.demo_name)
        filename = self.demo_name
        #open the file
        hf = h5py.File(filename, 'r')
        #navigate to necessary data and store in numpy arrays
        tf_info = hf.get('transform_info')
        js_info = hf.get('joint_state_info')
        pos_data = tf_info.get('transform_positions')
        rot_data = tf_info.get('transform_orientations')
        pos_data = np.array(pos_data)
        rot_data = np.array(rot_data)
        js_data = js_info.get('joint_positions')
        js_data = np.array(js_data)
        print("JS DATA SHAPE", np.shape(js_data))
        #close out file
        hf.close()
        
        
        #### move to starting position ####
        self.gripper_publisher = rospy.Publisher('/gripper_sends/position', Int32, queue_size=20)
        print('Going to Joint State: ' + str(js_data[0]))
        self.goto_joint_state(js_data, 0)
        
        print('start')
        print(self.repro_traj[0, :])
        print('end')
        print(self.repro_traj[-1, :])
        (n_pts, n_dims) = np.shape(self.repro_traj)
        
        
        self.goto_xyz(self.repro_traj, rot_data)

        
        (n_pts_og, _) = np.shape(rot_data)
        og_moments = [0.0, 0.25, 0.5, 0.75, 1.0]
        moments = [0.0, 0.25, 0.5, 0.75, 1.0]
        true_inds = [int(og_moments[i] * (n_pts_og - 1)) for i in range(len(og_moments))]
        key_inds = [int(moments[i] * (n_pts - 1)) for i in range(len(moments))]
        key_rots = R.from_quat([ [rot_data[ind][0], rot_data[ind][1], rot_data[ind][2], rot_data[ind][3]] for ind in true_inds])
        print(key_rots.as_quat())
        print(true_inds)
        print(key_inds)
        
        slerp = Slerp(key_inds, key_rots)
        
        #print('Press enter to continue')
        #input()
        print('Planning')
        waypoints = []
        wpose = self.move_group.get_current_pose().pose
        for i in range(1, n_pts):
        
            wpose.position.x = -self.repro_traj[i][0] #/tf and rviz have x and y opposite signs
            wpose.position.y = -self.repro_traj[i][1] 
            wpose.position.z = self.repro_traj[i][2]
          
            cur_R = slerp(np.array([i]))
            cur_quats = cur_R.as_quat()
            #print(cur_quats)
            wpose.orientation.x = -cur_quats[0][1]
            wpose.orientation.y = cur_quats[0][0]
            wpose.orientation.z = cur_quats[0][3]
            wpose.orientation.w = -cur_quats[0][2]
          
            waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = self.move_group.compute_cartesian_path(
                                           waypoints,   # waypoints to follow
                                           0.01,       # eef_step
                                           0.0)       # jump_threshold
        print("Planning for " + str(fraction * 100) + "% of trajectory acheived!")
        if fraction < 0.4:
        	print("WARNING: Not enough planned!")
        	#exit()
        return plan

    def display_trajectory(self, plan):
        #ask rviz to display the trajectory
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        self.display_trajectory_publisher.publish(display_trajectory);

    def execute_plan(self, plan):
        #execute given plan
        self.move_group.execute(plan, wait=True)
        
    def execute_plan_gripper(self, plan):
        #execute given plan
        filename = self.demo_name
        #open the file
        '''
        hf = h5py.File(filename, 'r')
        #navigate to necessary data and store in numpy arrays
        tf_info = hf.get('transform_info')
        js_info = hf.get('joint_state_info')
        pos_data = tf_info.get('transform_positions')
        rot_data = tf_info.get('transform_orientations')
        pos_data = np.array(pos_data)
        rot_data = np.array(rot_data)
        js_data = js_info.get('joint_positions')
        js_data = np.array(js_data)
        gr_info = hf.get('gripper_info')
        gr_data = gr_info.get('gripper_position')
        gr_data = np.array(gr_data)
        #close out file
        hf.close()
        gr_points = downsample_traj(gr_data, 200)
        '''
        
        joint_pos, tf_pos, tf_rot, gr_data = read_data_ds(filename, 200)
        gr_points = gr_data
        #print(gr_points)
        #print(plan)
        total_time = plan.joint_trajectory.points[-1].time_from_start.secs + plan.joint_trajectory.points[-1].time_from_start.nsecs * 1e-9
        p2p_time = total_time / len(plan.joint_trajectory.points)
        cur_time = 0.
        for i in range(len(plan.joint_trajectory.points)):
            cur_time_rospy = rospy.Time.from_sec(cur_time)
            plan.joint_trajectory.points[i].time_from_start = cur_time_rospy
            cur_time += p2p_time
        #print(end_time)
        self.move_group.execute(plan, wait=False)
        for i in range(len(gr_points)):
            self.gripper_publisher.publish(int(gr_points[i][0]))
            rospy.sleep(total_time / len(gr_points))

    def wait_for_state_update(self, box_name, box_is_known=False, box_is_attached=False, timeout=4):
        #either this times out and returns false or the object is found within the planning scene and returns true
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            attached_objects = self.scene.get_attached_objects([box_name])
            is_attached = len(attached_objects.keys()) > 0
            is_known = box_name in self.scene.get_known_object_names()
            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True
            rospy.sleep(0.1)
            seconds = rospy.get_time()
        return False


    def add_table(self, timeout=4):
        #define a box for the table below the robot
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = self.robot.get_planning_frame()
        #box origin (default = {0, 0, 0, 0, 0, 0, 0})
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.z = -0.15
        self.box_name1 = "table"
        #add box to planning scene and specify dimensions
        self.scene.add_box(self.box_name1, box_pose, size=(10, 10, 0.1))
        #wait for the box to be added in or to timeout
        return self.wait_for_state_update(self.box_name1, box_is_known=True, timeout=timeout)

    def add_wall(self, timeout=4):
        #Same as above with different dimensions
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = self.robot.get_planning_frame()
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.y = -0.3 # next to the robot
        self.box_name2 = "wall"
        self.scene.add_box(self.box_name2, box_pose, size=(10, 0.02, 10))
        return self.wait_for_state_update(self.box_name2, box_is_known=True, timeout=timeout)
     
    def remove_workspace(self, timeout=4):
        #remove each object from the planning scene, waiting for scene to update before moving on
        self.scene.remove_world_object(self.box_name1)
        self.wait_for_state_update(self.box_name1, box_is_attached=False, box_is_known=False, timeout=timeout)
        self.scene.remove_world_object(self.box_name2)
        return self.wait_for_state_update(self.box_name2, box_is_attached=False, box_is_known=False, timeout=timeout) 

    
def read_data(filename):
    hf = h5py.File(filename, 'r')
    print(list(hf.keys()))
    js_info = hf.get('joint_state_info')
    print(list(js_info.keys()))
    joint_pos = np.array(js_info.get('joint_positions'))
    tf_info = hf.get('transform_info')
    print(list(tf_info.keys()))
    tf_pos = np.array(tf_info.get('transform_positions'))
    tf_rot = np.array(tf_info.get('transform_orientations'))
    return joint_pos, tf_pos, tf_rot
    
def read_data_ds(filename, n=100):
    global joint_pos, tf_pos, tf_rot, gr_data
    hf = h5py.File(filename, 'r')
    print(list(hf.keys()))
    js_info = hf.get('joint_state_info')
    print(list(js_info.keys()))
    joint_pos = np.array(js_info.get('joint_positions'))
    tf_info = hf.get('transform_info')
    print(list(tf_info.keys()))
    tf_pos = np.array(tf_info.get('transform_positions'))
    tf_rot = np.array(tf_info.get('transform_orientations'))
    gr_info = hf.get('gripper_info')
    gr_data = gr_info.get('gripper_position')
    gr_data = np.array(gr_data)
    tf_pos, ds_inds = db_downsample_inds(tf_pos, n)
    joint_pos = joint_pos[ds_inds, :]
    tf_rot = tf_rot[ds_inds, :]
    gr_data = gr_data[ds_inds, :]
    return joint_pos, tf_pos, tf_rot, gr_data
    
def read_new_h5(fname):
    hf = h5py.File(fname, 'r')
    
    demo = np.array(hf.get("demo"))
    gr_data = np.array(hf.get("gripper_position"))
    F_normalized = np.array(hf.get("F_norm"))
    
    o1 = hf.get("O1_min")
    f1 = np.array(o1.get("F"))
    x1 = np.array(o1.get("X"))
    repro1 = np.array(o1.get("sol"))
    
    o2 = hf.get("O2_min")
    f2 = np.array(o2.get("F"))
    x2 = np.array(o2.get("X"))
    repro2 = np.array(o2.get("sol"))
    
    o3 = hf.get("O3_min")
    f3 = np.array(o3.get("F"))
    x3 = np.array(o3.get("X"))
    repro3 = np.array(o3.get("sol"))
    
    pref = hf.get("pref_min")
    fp = np.array(pref.get("F"))
    xp = np.array(pref.get("X"))
    reprop = np.array(pref.get("sol"))
    
    nonp = hf.get("non_pareto")
    fnp = np.array(nonp.get("F"))
    xnp = np.array(nonp.get("X"))
    repronp = np.array(nonp.get("sol"))
    
    return [[demo, gr_data, F_normalized], [f1, x1, repro1], [f2, x2, repro2], [f3, x3, repro3], [fp, xp, reprop], [fnp, xnp, repronp]]
    
    

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Trajectory Selection")
        self.setGeometry(100, 100, 1200, 400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)


class PlotCanvas(QWidget):
    def __init__(self, fig):
        super().__init__()

        self.fig = fig
        self.canvas = self.fig.canvas

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.canvas)
        
        
class ranking_UI(object):
    def __init__(self, filename, objective_names=[], rviz=False):
        
        self.new_fname = filename
        self.old_fname = '../h5_files/' + filename[24:-12] + filename[-12:].replace('_', ':')
        
        self.objective_names = objective_names
        
        [[demo, gr_data, F_normalized], [f1, x1, repro1], [f2, x2, repro2], [f3, x3, repro3], [fp, xp, reprop], [fnp, xnp, repronp]] = read_new_h5(filename)
        
        
        self.demo = demo
        self.gr_data = gr_data
        self.F_normalized = F_normalized
        self.names = ['min SSE', 'min ANG', 'min SMT', 'best', 'non-pareto']
        self.trajs = [repro1, repro2, repro3, reprop, repronp]
        self.fs = [f1, f2, f3, fp, fnp]
        self.xs = [x1, x2, x3, xp, xnp]
        
        self.num_trajs = len(self.trajs)
        
        new_names = []
        new_trajs = []
        new_fs = []
        new_xs = []
        order = np.random.permutation(self.num_trajs)
        for i in range(self.num_trajs):
            new_names.append(self.names[order[i]])
            new_trajs.append(self.trajs[order[i]])
            new_fs.append(self.fs[order[i]])
            new_xs.append(self.xs[order[i]])
        self.names = new_names
        self.trajs = new_trajs
        self.fs = new_fs
        self.xs = new_xs
            
        
        self.idx = 0
        self.n_pts, self.n_dims = np.shape(self.demo)
        
        self.rankings = [0] * self.num_trajs
        
        self.app = QApplication(sys.argv)
        self.rviz = rviz
        if self.rviz:
            print("MAKE SURE TERMINAL IS SOURCED")
            rospy.init_node('hri_interface', anonymous=True)
            self.move_group = MoveGroupPythonInterface()
            #self.move_group.add_table()
            #self.move_group.add_wall()
            self.move_group.set_demo(self.old_fname)
            self.mv = MyViz()
        self.app.setStyleSheet('.QLabel { font-size: 14pt;}')
        self.window = MainWindow()
        self.window.setFixedWidth(720)
        self.window.setFixedHeight(980)
        
    def make_window(self):
        self.outermost_layout = QVBoxLayout() 
        self.get_title()
        self.outermost_layout.addWidget(self.title_label)
        if self.rviz:
            self.outermost_layout.addWidget(self.mv)
            self.get_rviz_buttons()
            self.outermost_layout.addLayout(self.rviz_button_layout)
        self.get_traj_plot()
        self.outermost_layout.addWidget(self.canvas)
        self.get_stats_rankings()
        self.outermost_layout.addLayout(self.stats_rankings_layout)
        self.get_nav_buttons()
        self.outermost_layout.addLayout(self.nav_button_layout)
        
        self.window.central_widget.setLayout(self.outermost_layout)
        self.window.show()
        
    def update_ui(self):
        self.update_traj()
        self.update_stats_rankings()
        
    def get_title(self):
        self.title_label = QLabel()
        self.title_label.setText('Please Rank Trajectories') 
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 32pt;")
        
    def get_rviz_buttons(self):
        self.rviz_button_layout = QHBoxLayout()
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.preview)
        self.rviz_button_layout.addWidget(self.preview_button)
        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.execute)
        self.rviz_button_layout.addWidget(self.execute_button)
        
    def get_traj_plot(self):
        self.fig = plt.figure(figsize=(4, 3), dpi=100)
        self.canvas = PlotCanvas(self.fig)
        if self.n_dims == 2:
            self.ax = self.fig.add_subplot(111)
            self.ax.plot(self.demo[:, 0], self.demo[:, 1], 'k--', label="Demo")
            traj = self.trajs[self.idx]
            self.ax.plot(traj[:, 0], traj[:, 1], 'r', label="Repro")
        elif self.n_dims == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.plot(self.demo[:, 0], self.demo[:, 1], self.demo[:, 2], 'k--', label="Demo")
            traj = self.trajs[self.idx]
            self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r', label="Repro")
        self.ax.legend()
        self.canvas.canvas.draw()
        
    def update_traj(self):
        self.fig.clear()
        if self.n_dims == 2:
            self.ax = self.fig.add_subplot(111)
            self.ax.plot(self.demo[:, 0], self.demo[:, 1], 'k--', label="Demo")
            traj = self.trajs[self.idx]
            self.ax.plot(traj[:, 0], traj[:, 1], 'r', label="Repro")
        elif self.n_dims == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.plot(self.demo[:, 0], self.demo[:, 1], self.demo[:, 2], 'k--', label="Demo")
            traj = self.trajs[self.idx]
            self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r', label="Repro")
        self.ax.legend()
        self.canvas.canvas.draw()
    
    def get_stats_rankings(self):
        self.stats_rankings_layout = QHBoxLayout()
        self.desc_label = QLabel()
        self.desc_label.setText(self.get_traj_str())
        self.desc_label.setAlignment(Qt.AlignLeft)
        self.stats_rankings_layout.addWidget(self.desc_label)
        
        self.radio_layout = QVBoxLayout()
        self.ranking_label = QLabel()
        self.ranking_label.setText('Ranking (' + str(np.sum(np.array(self.rankings) > 0)) + "/5 Selected)") 
        self.ranking_label.setAlignment(Qt.AlignCenter)
        self.ranking_label.setStyleSheet("font-size: 16pt;")
        self.radio_layout.addWidget(self.ranking_label)
        self.radio_buttons = []
        for rank in range(1, self.num_trajs + 1):
            if rank == 1:
                self.radio_buttons.append(QRadioButton(str(rank) + " (best)"))
            elif rank == self.num_trajs:
                self.radio_buttons.append(QRadioButton(str(rank) + " (worst)"))
            else:
                self.radio_buttons.append(QRadioButton(str(rank)))
            self.radio_buttons[rank-1].toggled.connect(self.set_ranking)
            self.radio_layout.addWidget(self.radio_buttons[rank-1])
        self.stats_rankings_layout.addLayout(self.radio_layout)
        
    def update_stats_rankings(self):
        self.desc_label.setText(self.get_traj_str())
        self.ranking_label.setText('Ranking (' + str(np.sum(np.array(self.rankings) > 0)) + "/5 Selected)") 
        for i in range(self.num_trajs):
            self.radio_buttons[i].toggled.disconnect()
            self.radio_buttons[i].setAutoExclusive(False)
            self.radio_buttons[i].setChecked(False)
            self.radio_buttons[i].setAutoExclusive(True)
            self.radio_buttons[i].toggled.connect(self.set_ranking)
        if self.rankings[self.idx] != 0:
            self.radio_buttons[self.rankings[self.idx] - 1].setChecked(True)
            
    def get_nav_buttons(self):
        self.nav_button_layout = QHBoxLayout()
        self.previous_button = QPushButton("Previous")
        self.previous_button.clicked.connect(self.previous)
        self.nav_button_layout.addWidget(self.previous_button)
        self.quit_button = QPushButton("Save + Quit")
        self.quit_button.clicked.connect(self.quit)
        self.nav_button_layout.addWidget(self.quit_button)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next)
        self.nav_button_layout.addWidget(self.next_button)
        
    def get_traj_str(self):
        out_str = "Trajectory Number: " + str(self.idx + 1) + "\n"
        out_str += "Metric Evaluation:\n"
        obj_results = self.fs[self.idx]
        for i in range(len(obj_results)):
            out_str = out_str + self.objective_names[i] + ": "
            if obj_results[i] > (2/3):
                out_str = out_str + "bad (" + str(round((1-obj_results[i]) * 100)) + "%)\n"
            elif obj_results[i] < (1/3):
                out_str = out_str + "good (" + str(round((1-obj_results[i]) * 100)) + "%)\n"
            else:
                out_str = out_str + "OK (" + str(round((1-obj_results[i]) * 100)) + "%)\n"
        return out_str
    
    def preview(self):
        print("Preview Trajectory")
        traj = self.trajs[self.idx]
        self.move_group.set_repro(traj)
        plan = self.move_group.plan_cartesian_path()
        self.move_group.display_trajectory(plan)
        
    def execute(self):
        print("Execute Trajectory")
        traj = self.trajs[self.idx]
        self.move_group.set_repro(traj)
        plan = self.move_group.plan_cartesian_path()
        #self.move_group.display_trajectory(plan)
        self.move_group.execute_plan_gripper(plan)
        
    def previous(self):
        print("Previous")
        self.idx -= 1
        if self.idx < 0:
            self.idx += self.num_trajs
        print(self.idx)
        self.update_ui()
        
    def quit(self):
        print("Save + Quit")
        #np.savetxt("rankings" + str(args.user) + ".txt", np.array(self.rankings))
        data = {
            'user_num' : str(args.user),
            'filename' : self.new_fname,
            'order' : self.names,
            'ranking' : self.rankings
        }
        
        with open("rankings" + str(args.user) + ".json", 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        if self.rviz:
            pass #self.move_group.remove_workspace()
        self.window.close()
        
    def closeEvent(self, event):
        # Auto-save on window close
        self.quit()
        event.accept()
        
    def next(self):
        print("Next")
        self.idx += 1
        if self.idx >= self.num_trajs:
            self.idx -= self.num_trajs
        print(self.idx)
        self.update_ui()
        
    def set_ranking(self, value):
        print("Ranking Set " + str(value))
        selected_ranking = 0
        for i in range(self.num_trajs):
            if self.radio_buttons[i].isChecked():
                selected_ranking = i+1
        for i in range(self.num_trajs):
            if self.rankings[i] == selected_ranking:
                self.rankings[i] = 0
        self.rankings[self.idx] = selected_ranking
        print("Ranking Set as " + str(selected_ranking) + " for Traj " + str(self.idx))
        print(self.rankings)
        self.ranking_label.setText('Ranking (' + str(np.sum(np.array(self.rankings) > 0)) + "/5 Selected)") 
        
def testing_main():
    ui = ranking_UI(filename='../new_repros/new_reprorecorded_demo 2024-12-04 12_12_35.h5', objective_names=["Spatial Similarity", "Angular Similarity", "Smoothness"], rviz=True)
    ui.make_window()
    ui.app.exec_()
    
def main_exp():
    demos_list = os.listdir('../new_repros3')
    print(demos_list)
    demo_fname = '../new_repros3/' + demos_list[int(args.user)-1]
    ui = ranking_UI(filename=demo_fname, objective_names=["Spatial Similarity", "Angular Similarity", "Smoothness"], rviz=True)
    ui.make_window()
    ui.app.exec_()
        
if __name__ == '__main__':
    main_exp()

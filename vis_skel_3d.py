from math import sqrt
import argparse
import os
import numpy as np
import trimesh
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('dark')
from PIL import Image

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Path to dataset install')
    parser.add_argument('--subject', required=True, default='Subject_1')
    parser.add_argument('--action_name', required=True, default='open_liquid_soap')
    parser.add_argument('--seq_idx', required=True, default='1')
    parser.add_argument('--frame_idx', required=True, default=0, type=int)
    parser.add_argument(
        '--obj', required=True, choices=['liquid_soap', 'juice_bottle', 'milk', 'salt'])
    args = parser.parse_args()
    return args

def get_skeleton(sample, skel_root):
    skeleton_path = os.path.join(skel_root, sample['subject'],
                                 sample['action_name'], sample['seq_idx'],
                                 'skeleton.txt')
    print('Loading skeleton from {}'.format(skeleton_path))
    skeleton_vals = np.loadtxt(skeleton_path)
    # print ("DEBUG Skeleton_vals : ", skeleton_vals)
    print ("DEBUG Skeleton_vals : ", skeleton_vals.shape)
    # skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21,
    #                                         -1)[sample['frame_idx']]
    skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21,
                                            -1)
    # print ("DEBUG Skeleton : ", skeleton)
    print ("DEBUG Skeleton : ", skeleton.shape)

    # skeleton = skeleton[sample['frame_idx']]
    return skeleton

def get_joint(joint_names, skeleton):
    Joint = {}
    for idx, name in enumerate(joint_names):
        Joint[name] = skel[idx]
        print (name, " : ", Joint[name])
    return Joint

def get_finger_pos(hand_conf, finger_names):
    finger_pos = [np.empty((0, 3))] * len(finger_names)
    for i, finger in enumerate(finger_names):
        finger_pos[i] = np.vstack((finger_pos[i], hand_conf["Wrist"]))
        j_name = finger + "MCP"
        # print (j_name)
        finger_pos[i] = np.vstack((finger_pos[i], hand_conf[j_name]))
        j_name = finger + "PIP"
        # print (j_name)
        finger_pos[i] = np.vstack((finger_pos[i], hand_conf[j_name]))
        j_name = finger + "DIP"
        # print (j_name)
        finger_pos[i] = np.vstack((finger_pos[i], hand_conf[j_name]))
        j_name = finger + "TIP"
        # print (j_name)
        finger_pos[i] = np.vstack((finger_pos[i], hand_conf[j_name]))
        # print (finger_pos[i].shape)
    # print (finger_pos[0])
    # print (finger_pos[0][:, 0])
    # print (hand_conf["Wrist"])
    return finger_pos

def get_names():
    joint_names = ["Wrist", "TMCP", "IMCP", "MMCP", "RMCP", "PMCP", "TPIP",
            "TDIP", "TTIP", "IPIP", "IDIP", "ITIP", "MPIP", "MDIP", "MTIP",
            "RPIP", "RDIP", "RTIP", "PPIP", "PDIP", "PTIP"]
    finger_names = ["T", "I", "M", "R", "P"]
    global_joint_names = ["MCP", "PIP", "DIP", "TIP"]
    angle_names = ["MCP", "PIP", "DIP"]
    return joint_names, finger_names, global_joint_names, angle_names

def get_figure():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    return fig, ax

def draw_plane(ax, joints, clr='b', alpha=0.3):
    # These two vectors are in the plane
    p1 = joints[0]
    p2 = joints[1]
    p3 = joints[2]

    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    x = np.linspace(10, 53, 5)
    y = np.linspace(-60, 10, 5)
    X, Y = np.meshgrid(x, y)
    
    Z = (d - a * X - b * Y) / c

    ax.plot_surface(X, Y, Z, color=clr, alpha=alpha)

def draw_triangle(ax, joints, clr='b', alpha=0.3):
    collection = Poly3DCollection(joints, alpha=alpha, linewidths=0)
    collection.set_facecolor(clr)
    ax.add_collection3d(collection)

def draw_arc(ax, pt1, pt2, ptc):
    line_x = np.array([ptc, pt2])
    line_y = np.array([ptc, pt1])
    print (line_x)
    print (line_y)
    print (intersection_exists(line_x[0], line_x[1], line_x[0], 1))
    inter_points_1 = intersection_points(line_x[0], line_x[1], line_x[0], 10)
    inter_points_2 = intersection_points(line_y[0], line_y[1], line_y[0], 10)

    print (inter_points_1)
    print (inter_points_2)

    t = np.linspace(0, 1, 100)
    phi = np.pi / 10
    theta = t * phi 

    arc = (np.expand_dims((np.sin((1 - t) * phi) / np.sin(phi)), axis=0).T) * np.expand_dims(inter_points_1[0], axis=0) + (np.expand_dims((np.sin(t * phi) / np.sin(phi)), axis=0).T) * np.expand_dims(inter_points_2[0], axis=0)

    # print (arc.shape)
    # print (inter_points[0])
    # print (inter_points.shape)
    # print (arc)
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], linewidth=3, color='black')

def get_angle_plot(line1, line2, offset = 1, color = None, 
                   origin = [0,0], len_x_axis = 1, 
                   len_y_axis = 1):
    l1xy = line1.get_xydata()

    # Angle between line1 and x-axis
    slope1 = (l1xy[1][1] - l1xy[0][2]) / float(l1xy[1][0] - l1xy[0][0])
    angle1 = abs(math.degrees(math.atan(slope1))) # Taking only the positive angle
    l2xy = line2.get_xydata()
    # Angle between line2 and x-axis
    slope2 = (l2xy[1][3] - l2xy[0][4]) / float(l2xy[1][0] - l2xy[0][0])
    angle2 = abs(math.degrees(math.atan(slope2)))

    theta1 = min(angle1, angle2)
    theta2 = max(angle1, angle2)
    angle = theta2 - theta1
    if color is None:
        color = line1.get_color() # Uses the color of line 1 if color parameter is not passed.
    return Arc(origin, len_x_axis*offset, len_y_axis*offset, 
               0, theta1, theta2, color=color, 
               label = str(angle)+u"\u00b0")

def intersection_points(pt1, pt2, ptc, r):
    a = (pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2 + (pt2[2] - pt1[2]) ** 2
    b = -2 * ((pt2[0] - pt1[0]) * (ptc[0] - pt1[0]) + (pt2[1] - pt1[1]) * (ptc[1] - pt1[1]) + (pt2[2] - pt1[2]) * (ptc[2] - pt1[2]))
    c = (ptc[0] - pt1[0]) ** 2 + (ptc[1] - pt1[1]) ** 2 + (ptc[2] - pt1[2]) ** 2 - r ** 2

    t1 = (-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    t2 = (-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return np.array([pt1 + t1 * (pt2 - pt1), pt1 + t2 * (pt2 - pt1)])

def intersection_exists(pt1, pt2, ptc, r):
    a = (pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2 + (pt2[2] - pt1[2]) ** 2
    
    b = -2 * ((pt2[0] - pt1[0]) * (ptc[0] - pt1[0]) + (pt2[1] - pt1[1]) * (ptc[1] - pt1[1]) + (pt2[2] - pt1[2]) * (ptc[2] - pt1[2]))

    c = (ptc[0] - pt1[0]) ** 2 + (ptc[1] - pt1[1]) ** 2 + (ptc[2] - pt1[2]) ** 2 - r ** 2

    return (b ** 2 - a * c) > 0 

if __name__ == '__main__':
    args = parse_args()
    sample = {
        'subject': args.subject,
        'action_name': args.action_name,
        'seq_idx': args.seq_idx,
        'frame_idx': args.frame_idx,
        'object': args.obj
    }
    print('Loading sample {}'.format(sample))

    joint_names, finger_names, global_joint_names, angle_names = get_names()

    skeleton_root = os.path.join(args.root, 'Hand_pose_annotation_v1')
    skel_trajectory = get_skeleton(sample, skeleton_root)
    
    skel = skel_trajectory[0]
    hand_conf = get_joint(joint_names, skel)
    finger_pos = get_finger_pos(hand_conf, finger_names)

    fig, ax = get_figure()

    for finger in finger_pos:
        ax.plot(finger[:, 0], finger[:, 1], finger[:, 2])
        ax.scatter3D(finger[:, 0], finger[:, 1], finger[:, 2])

    plane_points = [[hand_conf["Wrist"],
                    hand_conf["IMCP"], 
                    hand_conf["MMCP"], 
                    hand_conf["RMCP"], 
                    hand_conf["PMCP"]]]
    # draw_plane(ax, plane_points, clr='b')
    draw_triangle(ax, plane_points, clr='b', alpha=0.5)

    plane_points = [[hand_conf["Wrist"],
                    hand_conf["IMCP"], 
                    hand_conf["TMCP"]]]
    # draw_plane(ax, plane_points, clr='r')
    draw_triangle(ax, plane_points, clr='g', alpha=0.5)

    plane_points = [[hand_conf["IMCP"],
                    hand_conf["TMCP"], 
                    hand_conf["TPIP"]]]
    # draw_plane(ax, plane_points, clr='r')
    draw_triangle(ax, plane_points, clr='r', alpha=0.5)

    draw_arc(ax, hand_conf["Wrist"], hand_conf["PPIP"], hand_conf["PMCP"])
    draw_arc(ax, hand_conf["Wrist"], hand_conf["RPIP"], hand_conf["RMCP"])
    draw_arc(ax, hand_conf["Wrist"], hand_conf["MPIP"], hand_conf["MMCP"])
    draw_arc(ax, hand_conf["Wrist"], hand_conf["IPIP"], hand_conf["IMCP"])
    draw_arc(ax, hand_conf["Wrist"], hand_conf["TPIP"], hand_conf["TMCP"])
    
    draw_arc(ax, hand_conf["TMCP"], hand_conf["IPIP"], hand_conf["IMCP"])
    draw_arc(ax, hand_conf["TMCP"], hand_conf["TDIP"], hand_conf["TPIP"])
    # angle_plot = get_angle_plot(line_1, line_2, 1)
    # angle_text = get_angle_text(angle_plot) 
    # # Gets the arguments to be passed to ax.text as a list to display the angle value besides the arc

    # ax.scatter3D(inter_points_1[:, 0], inter_points_1[:, 1], inter_points_1[:, 2], c='r')
    # ax.scatter3D(inter_points_2[:, 0], inter_points_2[:, 1], inter_points_2[:, 2], c='r')
    # ax.add_patch(angle_plot) # To display the angle arc
    # ax.text(*angle_text) # To display the angle value
    
    # plt.axis('off')
    # for angle in range(0, 360):
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(.001)
    # ax.set_aspect('equal')
    plt.show()

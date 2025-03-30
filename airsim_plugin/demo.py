import airsim

import numpy as np
import cv2
import os
import sys
import json

sys.path.append("..")

from airsim_settings import ObservationDirections

from utils.env_utils import SimState, getPoseAfterMakeAction, getPoseAfterMakeActions
from Model.utils.common import observations_to_image, append_text_to_image, generate_video, quaterionPose2RulerPose


def convert_airsim_pose(pose):
    assert len(pose) == 7, "The length of input pose must be 7"
    formatted_airsim_pose = airsim.Pose(
        position_val=airsim.Vector3r(
            pose[0],
            pose[1],
            pose[2]
        ),
        orientation_val=airsim.Quaternionr(
            x_val=pose[3],
            y_val=pose[4],
            z_val=pose[5],
            w_val=pose[6],
        )
    )
    return formatted_airsim_pose



def get_pano_observations(
        current_pose: airsim.Pose,
        tool: AirVLNSimulatorClientTool,
):
    # return pano rgb-d observation.
    # pano sequence:
    # front, slightly right, right, slightly back right, back,
    # slightly back left, left, slightly left
    actions = [3,3,3]
    pano_obs = []
    new_pose = current_pose
    for i in range(8):
        obs_responses = tool.getImageResponses_v2(camera_id='front_0')
        pano_obs.append(obs_responses[0][0])
        new_pose = getPoseAfterMakeActions(new_pose, actions)
        tool.setPoses([[new_pose]])

    return pano_obs

def get_pano_observations_v2(
        current_pose: airsim.Pose,
        tool: AirVLNSimulatorClientTool,
        scene_id=0
):
    # return pano rgb-d observation.
    # pano sequence:
    # front, slightly right, right, slightly back right, back,
    # slightly back left, left, slightly left
    actions = [3,3,3]
    pano_obs = []
    pano_pose = []
    new_pose = current_pose
    for i in range(8):
        pano_pose.append(np.array(
            [new_pose.position.x_val, new_pose.position.y_val, new_pose.position.z_val,
             new_pose.orientation.x_val,new_pose.orientation.y_val,new_pose.orientation.z_val,new_pose.orientation.w_val]))
        # print(f"new pose:{new_pose}")
        if scene_id in [1, 7]:
            obs_responses = tool.getImageResponses(camera_id='front_0')
        else:
            obs_responses = tool.getImageResponses_v2(camera_id='front_0')
        pano_obs.append(obs_responses[0][0])
        # print(pano_obs[-1][0].shape)
        new_pose = getPoseAfterMakeActions(new_pose, actions)
        tool.setPoses([[new_pose]])

    return pano_obs, pano_pose

def traj_visualize(trajectory_files, scene_id):

    # load env
    machines_info_xxx = [
        {
            'MACHINE_IP': '127.0.0.1',
            'SOCKET_PORT': 30000,
            'MAX_SCENE_NUM': 8,
            'open_scenes': [scene_id],
        },
    ]

    tool = AirVLNSimulatorClientTool(machines_info=machines_info_xxx)
    tool.run_call()

    with open(trajectory_files, 'r') as f:
        meta_data = json.load(f)
        meta_data = meta_data["episodes"]
    for i, traj_info in enumerate(meta_data):
        if traj_info["scene_id"] != scene_id:
            continue

        text_instruction = traj_info['instruction']
        traj_id = traj_info['trajectory_id']

        os.makedirs(os.path.join("./traj_obs", traj_id), exist_ok=True)

        start_position = traj_info['start_position']
        start_rotation = traj_info['start_rotation']
        action_list = traj_info['actions']
        curr_pose = convert_airsim_pose(start_position+start_rotation)

        tool.setPoses([[curr_pose]])

        for t, act in enumerate(action_list):
            pano_obs, pano_pose = get_pano_observations_v2(curr_pose, tool, scene_id=scene_id)
            pano_obs_imgs = [pano_obs[6][0], pano_obs[7][0], pano_obs[0][0], pano_obs[1][0], pano_obs[2][0],
                             pano_obs[4][0]]
            pano_obs_deps = [pano_obs[6][1], pano_obs[7][1], pano_obs[0][1], pano_obs[1][1], pano_obs[2][1],
                             pano_obs[4][1]]

            pano_obs_imgs_path = ["traj_obs/{}/rgb_obs_{}_{}.png".format(traj_id,view_drc.replace(" ", "_"), t) for view_drc in
                                  ObservationDirections + ["back"]]
            pano_obs_deps_path = ["traj_obs/{}/dep_obs_{}_{}.npy".format(traj_id, view_drc.replace(" ", "_"), t) for view_drc in
                                  ObservationDirections + ["back"]]
            pano_pose_path = ["traj_obs/{}/pose_{}_{}.npy".format(traj_id, view_drc.replace(" ", "_"), t) for view_drc in
                              ObservationDirections + ["back"]]
            # print(pano_obs_deps[0])

            for j in range(len(pano_obs_imgs_path)):
                cv2.imwrite(pano_obs_imgs_path[j], pano_obs_imgs[j])
                np.save(pano_obs_deps_path[j], pano_obs_deps[j])
                np.save(pano_pose_path[j], pano_pose[j])

                pano_obs_depvis = (pano_obs_deps[j].squeeze() * 255).astype(np.uint8)
                pano_obs_depvis = np.stack([pano_obs_depvis for _ in range(3)], axis=2)

                cv2.imwrite(pano_obs_deps_path[j].replace("npy", "png"), pano_obs_depvis)

            new_pose = getPoseAfterMakeActions(curr_pose, [act])
            tool.setPoses([[new_pose]])
            curr_pose = new_pose

        break


def random_walk_demo(max_step_size=60):

    # load env
    machines_info_xxx = [
        {
            'MACHINE_IP': '127.0.0.1',
            'SOCKET_PORT': 30000,
            'MAX_SCENE_NUM': 8,
            'open_scenes': [scene_id],
        },
    ]

    tool = AirVLNSimulatorClientTool(machines_info=machines_info_xxx)
    tool.run_call()

    # sr_cnt = 0.0
    # spl_cnt = 0.0
    # total_cnt = len(navi_tasks)
    # ne = []
    step_size = 0
    hist_act_codes = []
    total_mv_length = 0.0
    while step_size < max_step_size:
        print("step size: {}".format(step_size))
        # get observation
        try:
            pano_obs = get_pano_observations(curr_pose, tool)
            pano_obs_imgs = [pano_obs[6][0], pano_obs[7][0], pano_obs[0][0], pano_obs[1][0], pano_obs[2][0],
                             pano_obs[4][0]]
            pano_obs_deps = [pano_obs[6][1], pano_obs[7][1], pano_obs[0][1], pano_obs[1][1], pano_obs[2][1],
                             pano_obs[4][1]]

            pano_obs_imgs_path = ["obs_imgs/rgb_obs_{}.png".format(view_drc.replace(" ", "_")) for view_drc in
                                  ObservationDirections + ["back"]]
            pano_obs_deps_path = ["obs_imgs/dep_obs_{}.tiff".format(view_drc.replace(" ", "_")) for view_drc in
                                  ObservationDirections + ["back"]]
            # print(pano_obs_deps[0])

            for j in range(len(pano_obs_imgs_path)):
                cv2.imwrite(pano_obs_imgs_path[j], pano_obs_imgs[j])
                cv2.imwrite(pano_obs_deps_path[j], pano_obs_deps[j])

                pano_obs_depvis = (pano_obs_deps[j].squeeze() * 255).astype(np.uint8)
                pano_obs_depvis = np.stack([pano_obs_depvis for _ in range(3)], axis=2)

                cv2.imwrite(pano_obs_deps_path[j].replace("tiff", "png"), pano_obs_depvis)
        except:
            break

        act_code = np.random.randint(0, 6)
        hist_act_codes.append(act_code)
        if act_code == 0:
            break

        new_pose = getPoseAfterMakeActions(curr_pose, [act_code])
        tool.setPoses([[new_pose]])

        step_dist = np.linalg.norm(np.array(list(new_pose.position)) - np.array(list(curr_pose.position)))
        total_mv_length += step_dist

        curr_pose = new_pose
        step_size += 1


def manual_control_vis(scene_id):
    '''
    manual control the drone and save the trajectory
    :return:
    '''
    save_root = "/home/vincent/py-pro/AirVLN-main/DATA/data/aerialvln"
    save_file_path = os.path.join(save_root, f"random_walk_data_env_{scene_id}.json")

    frames = []
    trajectory = []
    action_list = []

    machines_info_xxx = [
        {
            'MACHINE_IP': '127.0.0.1',
            'SOCKET_PORT': 30000,
            'MAX_SCENE_NUM': 8,
            'open_scenes': [scene_id],
        },
    ]

    tool = AirVLNSimulatorClientTool(machines_info=machines_info_xxx)
    tool.run_call()

    formatted_initial_pose = airsim.Pose(
        position_val=airsim.Vector3r(
            47.84111429433478,
            -17.650212810420353,
            -4.0
        ),
        orientation_val=airsim.Quaternionr(
            x_val=0.0,
            y_val=0.0,
            z_val=0,
            w_val=0
        )
    )

    formatted_initial_pose = convert_airsim_pose([-45.40736984755401, 7.5118277639380375, -6.713590621948242, 0.0, 0.0, -0.39943134648758677, 0.9167631097743372])
    tool.setPoses([[formatted_initial_pose]])
    current_pose = formatted_initial_pose

    trajectory.append(quaterionPose2RulerPose(current_pose))

    action_pre_list = []
    for a in action_pre_list:
        new_pose = getPoseAfterMakeAction(current_pose, a)
        tool.setPoses([[new_pose]])
        current_pose = new_pose
        trajectory.append(quaterionPose2RulerPose(current_pose))

    while True:
        print("current pose:", current_pose)
        print("action list:", action_pre_list+action_list)

        print("formatted current pose:", list(current_pose.position)+list(current_pose.orientation))

        obs_responses = tool.getImageResponses(camera_id='front_0')
        img_rgb = obs_responses[0][0][0]
        img_dep = obs_responses[0][0][1]

        obs = {"rgb": img_rgb, "depth": img_dep}
        frame = observations_to_image(obs, None)
        frame = append_text_to_image(frame, 'ramdom_walk')
        frames.append(frame)

        cv2.imshow("img", frame)
        cv2.waitKey()
        cv2.destroyAllWindows()

        action = input("""
        action list:
        0: STOP,
        1: MOVE FORWARD,
        2: TURN LEFT,
        3: TURN RIGHT,
        4: GO UP,
        5: GO DOWN,
        6: MOVE LEFT,
        7: MOVE RIGHT,

        Your input action number is: 
        """)
        action = int(action)
        new_pose = getPoseAfterMakeAction(current_pose, action)
        tool.setPoses([[new_pose]])
        current_pose = new_pose
        trajectory.append(quaterionPose2RulerPose(current_pose))
        action_list.append(action)
        if action == 0:
            break

    if not os.path.exists(save_file_path):
        with open(save_file_path, "w") as f:
            data = {"data": []}
    else:
        with open(save_file_path, "r") as f:
            data = json.load(f)

    data["data"].append(
        {
            "original_id": f"random{len(data['data'])}",
            "instruction": f"random walk{len(data['data'])}",
            "trajectory": trajectory
        }
    )

    with open(save_file_path, 'w') as f:
        json.dump(data, f)

    # if len(frames) > 0 and visualize:
    #     h, w = frames[0].shape[:2]
    #     fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    #     out = cv2.VideoWriter(
    #         '/home/vincent/py-pro/AirVLN-main/AirVLN/files/videos/output_manual_{}.avi'.format(episode_id), fourcc, 1,
    #         (w, h))
    #
    #     for frame in frames:
    #         out.write(frame)
    #
    #     out.release()
    #     print("Video processing complete.")


if __name__ == "__main__":
    random_walk_demo()


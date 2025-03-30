import msgpackrpc
import time
import airsim
import threading
import random
import copy
import numpy as np
import cv2
import os
import sys

sys.path.append("..")
from src.common.param import args

from airsim_settings import ObservationDirections

from utils.logger import logger
from utils.env_utils import SimState, getPoseAfterMakeAction, getPoseAfterMakeActions


class MyThread(threading.Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.flag_ok = False

    def run(self):
        try:
            self.result = self.func(*self.args)
        except Exception as e:
            logger.error(e)
            self.flag_ok = False
        else:
            self.flag_ok = True

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except:
            return None


class AirVLNSimulatorClientTool:
    def __init__(self, machines_info) -> None:
        self.machines_info = copy.deepcopy(machines_info)
        self.socket_clients = []
        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in machines_info ]

        self._init_check()

    def _init_check(self) -> None:
        ips = [item['MACHINE_IP'] for item in self.machines_info]
        assert len(ips) == len(set(ips)), 'MACHINE_IP repeat'

    def _confirmSocketConnection(self, socket_client: msgpackrpc.Client) -> bool:
        try:
            socket_client.call('ping')
            print("Connected\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            return True
        except:
            try:
                print("Ping returned false\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            except:
                print('Ping returned false')
            return False

    def _confirmConnection(self) -> None:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    self.airsim_clients[index_1][index_2].confirmConnection()

        return

    def _closeSocketConnection(self) -> None:
        socket_clients = self.socket_clients

        for socket_client in socket_clients:
            try:
                socket_client.close()
            except Exception as e:
                pass

        self.socket_clients = []
        return

    def _closeConnection(self) -> None:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    try:
                        self.airsim_clients[index_1][index_2].close()
                    except Exception as e:
                        pass

        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in self.machines_info]
        return

    def run_call(self, airsim_timeout: int=60) -> None:
        socket_clients = []
        for index, item in enumerate(self.machines_info):
            socket_clients.append(
                msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=180)
            )

        for socket_client in socket_clients:
            if not self._confirmSocketConnection(socket_client):
                logger.error('cannot establish socket')
                raise Exception('cannot establish socket')

        self.socket_clients = socket_clients


        before = time.time()
        self._closeConnection()

        def _run_command(index, socket_client: msgpackrpc.Client):
            logger.info(f'开始打开场景，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
            result = socket_client.call('reopen_scenes', socket_client.address._host, self.machines_info[index]['open_scenes'])

            print(result)
            if result[0] == False:
                logger.error(f'打开场景失败，机器: {socket_client.address._host}:{socket_client.address._port}')
                raise Exception('打开场景失败')
            assert len(result[1]) == 2, '打开场景失败'

            ip = result[1][0]
            ports = result[1][1]
            assert ip.decode("utf-8") == str(socket_client.address._host), '打开场景失败'
            assert len(ports) == len(self.machines_info[index]['open_scenes']), '打开场景失败'
            for i, port in enumerate(ports):
                if self.machines_info[index]['open_scenes'][i] is None:
                    self.airsim_clients[index][i] = None
                else:
                    self.airsim_clients[index][i] = airsim.VehicleClient(ip=ip, port=port, timeout_value=airsim_timeout)

            logger.info(f'打开场景完毕，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
            return

        threads = []
        thread_results = []
        for index, socket_client in enumerate(socket_clients):
            threads.append(
                MyThread(_run_command, (index, socket_client))
            )
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
        for thread in threads:
            thread.join()
        for thread in threads:
            thread.get_result()
            thread_results.append(thread.flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            raise Exception('打开场景失败')

        after = time.time()
        diff = after - before
        logger.info(f"启动时间：{diff}")

        self._confirmConnection()
        self._closeSocketConnection()

    def getImageResponses(self, get_rgb=True, get_depth=True, camera_id='front_0'):

        def _getImages(airsim_client: airsim.VehicleClient, scen_id, get_rgb, get_depth, camera_id='front_0'):
            if airsim_client is None:
                raise Exception('error')
                return None, None

            img_rgb = None
            img_depth = None

            if not get_rgb and not get_depth:
                return None, None

            if scen_id in [1, 7]:
                time_sleep_cnt = 0
                while True:
                    try:
                        ImageRequest = []
                        if get_rgb:
                            ImageRequest.append(
                                airsim.ImageRequest(camera_id, airsim.ImageType.Scene, pixels_as_float=False, compress=False)
                            )
                        if get_depth:
                            ImageRequest.append(
                                airsim.ImageRequest(camera_id, airsim.ImageType.DepthVis, pixels_as_float=False, compress=True)
                            )

                        responses = airsim_client.simGetImages(ImageRequest, vehicle_name='Drone_1')

                        if get_rgb and get_depth:
                            response_rgb = responses[0]
                            response_depth = responses[1]
                        elif get_rgb and not get_depth:
                            response_rgb = responses[0]
                        elif not get_rgb and get_depth:
                            response_depth = responses[0]
                        else:
                            break


                        img_rgb = None
                        img_depth = None

                        if get_rgb:
                            assert response_rgb.height == args.Image_Height_RGB and response_rgb.width == args.Image_Width_RGB, 'RGB图片size inconsistent'

                            img1d = np.frombuffer(response_rgb.image_data_uint8, dtype=np.uint8)
                            if args.run_type not in ['eval']:
                                assert not (img1d.flatten()[0] == img1d).all(), 'RGB图片获取错误'
                            img_rgb = img1d.reshape(response_rgb.height, response_rgb.width, 3)
                            img_rgb = np.array(img_rgb)

                        if get_depth:
                            assert response_depth.height == args.Image_Height_DEPTH and response_depth.width == args.Image_Width_DEPTH, 'DEPTH图片size inconsistent'

                            png_file_name = '/tmp/AirVLN_depth_{}_{}.png'.format(time.time(), random.randint(0, 10000))
                            airsim.write_file(png_file_name, response_depth.image_data_uint8)
                            img3d = cv2.imread(png_file_name)

                            os.remove(png_file_name)

                            img1d = img3d[:, :, 1]
                            img1d = img1d.reshape(response_depth.height, response_depth.width, 1)

                            obs_depth_img = img1d / 255

                            img_depth = np.array(obs_depth_img, dtype=np.float32)

                        break
                    except:
                        time_sleep_cnt += 1
                        logger.error("图片获取错误")
                        logger.error('time_sleep_cnt: {}'.format(time_sleep_cnt))
                        time.sleep(1)

                    if time_sleep_cnt > 20:
                        raise Exception('图片获取失败')

            else:
                time_sleep_cnt = 0
                while True:
                    try:
                        ImageRequest = []
                        if get_rgb:
                            ImageRequest.append(
                                airsim.ImageRequest(camera_id, airsim.ImageType.Scene, pixels_as_float=False, compress=False)
                            )
                        if get_depth:
                            ImageRequest.append(
                                airsim.ImageRequest(camera_id, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)
                            )

                        responses = airsim_client.simGetImages(ImageRequest, vehicle_name='Drone_1')

                        if get_rgb and get_depth:
                            response_rgb = responses[0]
                            response_depth = responses[1]
                        elif get_rgb and not get_depth:
                            response_rgb = responses[0]
                        elif not get_rgb and get_depth:
                            response_depth = responses[0]
                        else:
                            break

                        if get_rgb:
                            assert response_rgb.height == args.Image_Height_RGB and response_rgb.width == args.Image_Width_RGB, 'RGB图片获取错误'

                            img1d = np.frombuffer(response_rgb.image_data_uint8, dtype=np.uint8)
                            img_rgb = img1d.reshape(response_rgb.height, response_rgb.width, 3)
                            img_rgb = np.array(img_rgb)

                        if get_depth:
                            assert response_depth.height == args.Image_Height_DEPTH and response_depth.width == args.Image_Width_DEPTH, 'DEPTH图片获取错误'

                            depth_img_in_meters = airsim.list_to_2d_float_array(response_depth.image_data_float, response_depth.width, response_depth.height)
                            if depth_img_in_meters.min() < 1e4:
                                assert not (depth_img_in_meters.flatten()[0] == depth_img_in_meters).all(), 'DEPTH图片获取错误'
                            depth_img_in_meters = depth_img_in_meters.reshape(response_depth.height, response_depth.width, 1)

                            obs_depth_img = np.clip(depth_img_in_meters, 0, 100)
                            obs_depth_img = obs_depth_img / 100
                            # obs_depth_img = depth_img_in_meters
                            img_depth = np.array(obs_depth_img, dtype=np.float32)

                        break
                    except:
                        time_sleep_cnt += 1
                        logger.error("图片获取错误")
                        logger.error('time_sleep_cnt: {}'.format(time_sleep_cnt))
                        time.sleep(1)

                    if time_sleep_cnt > 20:
                        raise Exception('图片获取失败')

            return img_rgb, img_depth

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(
                        _getImages,
                        (
                            self.airsim_clients[index_1][index_2],
                            self.machines_info[index_1]['open_scenes'][index_2],
                            get_rgb, get_depth, camera_id))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()

        responses = []
        for index_1, _ in enumerate(threads):
            responses.append([])
            for index_2, _ in enumerate(threads[index_1]):
                responses[index_1].append(
                    threads[index_1][index_2].get_result()
                )
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('getImageResponses失败')
            return None

        return responses


    def getImageResponses_v2(self, get_rgb=True, get_depth=True, camera_id='front_0'):

        def _getImages(airsim_client: airsim.VehicleClient, scen_id, get_rgb, get_depth, camera_id='front_0'):
            if airsim_client is None:
                raise Exception('error')
                return None, None

            img_rgb = None
            img_depth = None

            if not get_rgb and not get_depth:
                return None, None

            time_sleep_cnt = 0
            while True:
                try:
                    ImageRequest = []
                    if get_rgb:
                        ImageRequest.append(
                            airsim.ImageRequest(camera_id, airsim.ImageType.Scene, pixels_as_float=False, compress=False)
                        )
                    if get_depth:
                        ImageRequest.append(
                            airsim.ImageRequest(camera_id, airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)
                        )

                    responses = airsim_client.simGetImages(ImageRequest, vehicle_name='Drone_1')

                    if get_rgb and get_depth:
                        response_rgb = responses[0]
                        response_depth = responses[1]
                    elif get_rgb and not get_depth:
                        response_rgb = responses[0]
                    elif not get_rgb and get_depth:
                        response_depth = responses[0]
                    else:
                        break

                    if get_rgb:
                        assert response_rgb.height == args.Image_Height_RGB and response_rgb.width == args.Image_Width_RGB, 'RGB图片获取错误'

                        img1d = np.frombuffer(response_rgb.image_data_uint8, dtype=np.uint8)
                        img_rgb = img1d.reshape(response_rgb.height, response_rgb.width, 3)
                        img_rgb = np.array(img_rgb)

                    if get_depth:
                        assert response_depth.height == args.Image_Height_DEPTH and response_depth.width == args.Image_Width_DEPTH, 'DEPTH图片获取错误'

                        depth_img_in_meters = airsim.list_to_2d_float_array(response_depth.image_data_float, response_depth.width, response_depth.height)
                        if depth_img_in_meters.min() < 1e4:
                            assert not (depth_img_in_meters.flatten()[0] == depth_img_in_meters).all(), 'DEPTH图片获取错误'
                        depth_img_in_meters = depth_img_in_meters.reshape(response_depth.height, response_depth.width, 1)

                        obs_depth_img = np.clip(depth_img_in_meters, 0, 100)
                        obs_depth_img = obs_depth_img / 100
                        # obs_depth_img = depth_img_in_meters
                        img_depth = np.array(obs_depth_img, dtype=np.float32)

                    break
                except:
                    time_sleep_cnt += 1
                    logger.error("图片获取错误")
                    logger.error('time_sleep_cnt: {}'.format(time_sleep_cnt))
                    time.sleep(1)

                if time_sleep_cnt > 20:
                    raise Exception('图片获取失败')

            return img_rgb, img_depth

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(
                        _getImages,
                        (
                            self.airsim_clients[index_1][index_2],
                            self.machines_info[index_1]['open_scenes'][index_2],
                            get_rgb, get_depth, camera_id))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()

        responses = []
        for index_1, _ in enumerate(threads):
            responses.append([])
            for index_2, _ in enumerate(threads[index_1]):
                responses[index_1].append(
                    threads[index_1][index_2].get_result()
                )
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('getImageResponses失败')
            return None

        return responses


    def setPoses(self, poses: list) -> bool:
        def _setPoses(airsim_client: airsim.VehicleClient, pose: airsim.Pose) -> None:
            if airsim_client is None:
                raise Exception('error')
                return

            airsim_client.simSetVehiclePose(
                pose=pose,
                ignore_collision=True,
                vehicle_name='Drone_1',
            )

            return

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_setPoses, (self.airsim_clients[index_1][index_2], poses[index_1][index_2]))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].get_result()
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('setPoses失败')
            return False

        return True

    def closeScenes(self):
        try:
            socket_clients = []
            for index, item in enumerate(self.machines_info):
                socket_clients.append(
                    msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=180)
                )

            for socket_client in socket_clients:
                if not self._confirmSocketConnection(socket_client):
                    logger.error('cannot establish socket')
                    raise Exception('cannot establish socket')

            self.socket_clients = socket_clients


            self._closeConnection()

            def _run_command(index, socket_client: msgpackrpc.Client):
                logger.info(f'开始关闭所有场景，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
                result = socket_client.call('close_scenes', socket_client.address._host)
                logger.info(f'关闭所有场景完毕，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
                return

            threads = []
            for index, socket_client in enumerate(socket_clients):
                threads.append(
                    MyThread(_run_command, (index, socket_client))
                )
            for thread in threads:
                thread.setDaemon(True)
                thread.start()
            for thread in threads:
                thread.join()
            threads = []

            self._closeSocketConnection()
        except Exception as e:
            logger.error(e)


def visualize_demo():
    machines_info_xxx = [
        {
            'MACHINE_IP': '127.0.0.1',
            'SOCKET_PORT': 30000,
            'MAX_SCENE_NUM': 8,
            'open_scenes': [1],
        },
    ]

    tool = AirVLNSimulatorClientTool(machines_info=machines_info_xxx)
    tool.run_call()

    start_time = time.time()
    while True:
        time_1 = time.time()
        responses = tool.getImageResponses(camera_id='bottom_0')
        time_2 = time.time()
        print(
            "total_time: {} \t time: {} \t fps: {}".format(
                (time_2-start_time),
                (time_2-time_1),
                1/(time_2-time_1),
            )
        )

        img = responses[0][0][0]
        print(img.shape)
        cv2.imshow("img", img)
        cv2.waitKey()
        cv2.imwrite("../files/{}.png".format(time.time()), img)

        poses = []
        for index_1, item in enumerate(machines_info_xxx):
            poses.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                pose=airsim.Pose(
                    position_val=airsim.Vector3r(random.randint(0, 100), random.randint(0, 100), -30),
                    # position_val=airsim.Vector3r(0, 0, -30),
                    orientation_val=airsim.Quaternionr(0, 0, 0, 1),
                )
                poses[index_1].append(pose)

        tool.setPoses(poses)


def get_pano_observations(
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
        print(f"new pose:{new_pose}")
        if scene_id in [1, 7]:
            obs_responses = tool.getImageResponses(camera_id='front_0')
        else:
            obs_responses = tool.getImageResponses_v2(camera_id='front_0')
        pano_obs.append(obs_responses[0][0])
        print(pano_obs[-1][0].shape)
        new_pose = getPoseAfterMakeActions(new_pose, actions)
        tool.setPoses([[new_pose]])

    return pano_obs, pano_pose


def prompt_updator(original_prompt, ongoing_task=None, action_code=None, observations=None, action_seq_num=1):
    import re
    ori_prompt_splits = original_prompt.split("\n")

    action_seq = "\n".join(ori_prompt_splits[14:-3])

    action_str = ""
    if action_code == 0:
        action_str = "STOP"
    elif action_code == 1:
        action_str = "MOVE FORWARD"
    elif action_code == 2:
        action_str = "TURN LEFT"
    elif action_code == 3:
        action_str = "TURN RIGHT"
    elif action_code == 4:
        action_str = "GO UP"
    elif action_code == 5:
        action_str = "GO DOWN"

    if action_str != "":
        pattern = re.compile(r'^\d+')
        action_seq_num = 0
        for i in range(len(ori_prompt_splits)-1, -1, -1):
            m = re.match(pattern, ori_prompt_splits[i])
            if m:
                action_seq_num = int(m.group()) + 1
                break
        if not action_seq_num:
            action_seq_num = 1

        if action_seq != "":
            action_seq = f"{action_seq}\n{action_seq_num}. {action_str}\n"
        else:
            action_seq = f"{action_seq_num}. {action_str}\n"

    observation_str = ""
    if observations:
        for landmark in observations:
            coarse_grained_loc, fine_grained_loc = observations[landmark]
            landmark_obs_str = f"There is {landmark} on the {fine_grained_loc} side of your {coarse_grained_loc} view.\n"
            observation_str += landmark_obs_str

    if observation_str != "":
        action_seq += observation_str

    navi_task = ori_prompt_splits[12]
    if ongoing_task:
        navi_task = ongoing_task

    prompt = f"""
You are a drone and your task is navigating to the described target location!
Action Space: MOVE FORWARD, TURN LEFT, TURN RIGHT, GO UP, GO DOWN, STOP . The angle you turn left or right each time is 15 degrees.

Observation Direction: 
front: facing directly to the front
left: facing directly to the left
right: facing directly to the right
slightly left: 45 degrees to the front left
slightly right: 45 degrees to the front right

Navigation Instructions:
{navi_task}
Action Sequence:
{action_seq}
<Your next immediate action, and your reason>
    """

    # print(prompt)
    return prompt


def landmark_observation_gen(landmarks, landmark_scores, landmark_bboxes, img_size):
    assert len(landmark_scores) == len(landmark_bboxes)
    assert len(landmark_scores) == len(landmarks)

    image_height, image_width = img_size

    landmark_obs = {}
    for i, landmark in enumerate(landmarks):
        scores = [0 if x is None else x for x in landmark_scores[i]]
        idx = scores.index(max(scores))
        score = landmark_scores[i][idx]
        bbox  = landmark_bboxes[i][idx]      # xyxy format, xy are in opencv coordinate
        if bbox is None:
            continue
        bbox_center = [0.5*(bbox[0]+bbox[2]), 0.5*(bbox[1]+bbox[3])]

        coarse_grained_loc = ObservationDirections[idx]
        fine_grained_loc   = ""
        if bbox_center[0] < image_width//3 and bbox_center[1] < image_height//3:
            fine_grained_loc = "top left"
        elif image_width//3 < bbox_center[0] < image_width*2//3 and bbox_center[1] < image_height//3:
            fine_grained_loc = "top center"
        elif image_width*2//3 < bbox_center[0] < image_width and bbox_center[1] < image_height//3:
            fine_grained_loc = "top right"
        elif bbox_center[0] < image_width//3 and image_height//3 < bbox_center[1] < image_height*2//3:
            fine_grained_loc = "left"
        elif image_width//3 < bbox_center[0] < image_width*2//3 and image_height//3 < bbox_center[1] < image_height*2//3:
            fine_grained_loc = "center"
        elif image_width*2//3 < bbox_center[0] < image_width and image_height//3 < bbox_center[1] < image_height*2//3:
            fine_grained_loc = "right"
        elif bbox_center[0] < image_width//3 and image_height*2//3 < bbox_center[1] < image_height:
            fine_grained_loc = "bottom left"
        elif image_width//3 < bbox_center[0] < image_width*2//3 and image_height*2//3 < bbox_center[1] < image_height:
            fine_grained_loc = "bottom center"
        elif image_width*2//3 < bbox_center[0] < image_width and image_height*2//3 < bbox_center[1] < image_height:
            fine_grained_loc = "bottom right"

        landmark_obs[landmark] = [coarse_grained_loc, fine_grained_loc]

    return landmark_obs


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


if __name__ == '__main__':
    visualize_demo()


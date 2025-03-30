import io
import json
import pickle
import torch
import msgpackrpc
import time
import airsim
import threading
import random
import copy
import numpy as np
import cv2
import os
import PIL
from PIL import Image
import networkx as nx

from scipy.spatial import distance
from tqdm import tqdm
from typing import List

from torchvision.ops import box_convert

# sys.path.append("..")
from src.common.param import args
from src.llm.query_llm import OpenAI_LLM_v3
from src.llm.prompt_builder import landmark_caption_prompt_builder, route_planning_prompt_builder, prompt_updator_v2, \
    action_parser, reformat_dino_prompt

from airsim_plugin.airsim_settings import ObservationDirections

from utils.logger import logger
from utils.env_utils import getPoseAfterMakeActions
from utils.maps import build_semantic_map, visualize_semantic_point_cloud, update_camera_pose,\
    convert_global_pc, statistical_filter, find_closest_node

from utils.utils import calculate_movement_steps

from Grounded_Sam_Lite.groundingdino.util.inference import load_model, predict
import Grounded_Sam_Lite.groundingdino.datasets.transforms as T
from Grounded_Sam_Lite.grounded_sam_api import GroundedSam

from lm_nav.navigation_graph import NavigationGraph
from lm_nav import pipeline

from scipy.spatial.transform import Rotation as R
from evaluator.nav_evaluator import CityNavEvaluator

from datasets.airvln_e import AirVLNDataLoader

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


def parse_action_response(query_resp):
    res = query_resp.strip("\n").strip(" ").lstrip("0123456789. ")
    action_code = None
    if "STOP" in res:
        action_code = 0
    elif "MOVE FORWARD" in res or "MOVE_FORWARD" in res:
        action_code = 1
    elif "TURN LEFT" in res or "TURN_LEFT" in res:
        action_code = 2
    elif "TURN RIGHT" in res or "TURN_RIGHT" in res:
        action_code = 3
    elif "GO UP" in res or "GO_UP" in res:
        action_code = 4
    elif "GO DOWN" in res or "GO_DOWN" in res:
        action_code = 5

    return action_code


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


def semantic_map_grounding(
        vlm,
        rgb_imgs: List[np.ndarray],
        dep_imgs: List[np.ndarray],
        cur_pose: np.ndarray,
        caption: str,
        visulization=False
) -> (np.ndarray, np.ndarray):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    pcs = []
    lms = []
    lds = []

    merged_pc = None
    merged_lm = None
    merged_ld = {"None": 0}

    for i in range(len(rgb_imgs)):
        image = rgb_imgs[i]
        depth = dep_imgs[i].squeeze()

        h, w, _ = image.shape
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image, _ = transform(image, None)

        boxes, logits, phrases = predict(
            model=vlm,
            image=image,
            caption=caption,
            box_threshold=0.35,
            text_threshold=0.3
        )

        bboxes = boxes * torch.Tensor([w, h, w, h])
        bboxes = box_convert(bboxes, in_fmt='cxcywh', out_fmt='xyxy').numpy()

        rot_rad = (i-2)*np.pi/4
        new_cam_pose = update_camera_pose(cur_pose, rot_rad)
        pc, lm, ld = build_semantic_map(depth, 90, new_cam_pose, bboxes, phrases)
        # visualize_semantic_point_cloud(pc, lm)

        pcs.append(pc)
        lms.append(lm)
        lds.append(ld)

        # uniform class
        for cls in ld:
            if cls not in merged_ld:
                new_l = len(merged_ld)
                merged_ld[cls] = new_l

    # uniform label
    for i in range(len(lms)):
        lm = lms[i]
        ld = lds[i]

        rep = {}
        for cls in ld:
            if cls in merged_ld:
                rep[ld[cls]] = merged_ld[cls]

        v_rep = np.vectorize(rep.get)
        lm = v_rep(lm)

        lms[i] = lm

    merged_pc = np.concatenate(pcs, axis=0)
    merged_lm = np.concatenate(lms, axis=0)
    # print(merged_pc.shape)

    if visulization:
        visualize_semantic_point_cloud(merged_pc, merged_lm)

    return merged_pc, merged_lm, merged_ld


def explore_pipeline_by_dino(
        curr_pose,
        llm, vlm,
        image_path: List[str],
        rgb_imgs: List[np.ndarray],
        dep_imgs: List[np.ndarray],
        navigation_instruction: str,
        scene_objects: List[str], landmarks_route: List[str]
):
    # image caption
    time1 = time.time()
    observed_obj = set()
    caption_prompt = landmark_caption_prompt_builder(scene_objects)
    for img_p in image_path:
        caption_res_str = llm.query_api(caption_prompt, image_path=img_p, show_response=False)
        obs_strs = caption_res_str.split(".")
        for o in obs_strs:
            if o.strip(" ") not in observed_obj:
                observed_obj.add(o)
    obs_obj_str = ".".join(list(observed_obj))
    # print(f"scene caption time: {time.time()-time1}")
    # obs_obj_str = 'billboard.antenna next to the bulletin.road. woods. wire pole'
    time1 = time.time()
    route_predict_prompt = route_planning_prompt_builder(obs_obj_str, navigation_instruction, landmarks_route[0])
    print("################### WARNING: route_planning_prompt_builder use landmark start from 0")
    route_predicted = llm.query_api(route_predict_prompt, show_response=False)

    # print(f"query time: {time.time()-time1}")
    print("route_predict_prompt: ", route_predict_prompt)
    print("route_predicted: ", route_predicted)

    # route point prediction
    cur_pos = np.array(list(curr_pose.position))
    cur_ori = np.array([curr_pose.orientation.x_val, curr_pose.orientation.y_val, curr_pose.orientation.z_val, curr_pose.orientation.w_val])
    cur_pose= np.concatenate([cur_pos, cur_ori], axis=0)

    # image grounding
    time1 = time.time()
    semantic_map, semantic_label, semantic_cls = \
        semantic_map_grounding(vlm, rgb_imgs, dep_imgs, cur_pose, route_predicted, visulization=False)

    # convert semantic map to airsim coordinate
    cam2ego_rot = np.array([[0, 0, 1.0],
                        [1.0, 0, 0],
                        [0, 1.0, 0]])
    ego2world_rot = R.from_quat(list(curr_pose.orientation)).as_matrix()
    coord_rot   = ego2world_rot.dot(cam2ego_rot)
    coord_trans = np.array(list(curr_pose.position)).reshape(-1, 1)
    semantic_map = (coord_rot.dot(semantic_map.T) + coord_trans).T      # n*3 in world coord system
    # print(f"semantic map construction time: {time.time() - time1}")

    time1 = time.time()

    routes = route_predicted.split(".")
    if routes[0].strip(" ") not in semantic_cls:
        route_coords = cur_pos
        # todo: ramdom walk
        pass
    else:
        next_route_label = semantic_cls[routes[0].strip(" ")]
        route_semantic_map = semantic_map[semantic_label.ravel()==next_route_label]
        route_coords = np.mean(route_semantic_map, axis=0)     # (3,)
        z_coord = cur_pos[2]
        alpha = 0.6
        route_coords = alpha * route_coords + (1-alpha) * cur_pos
        # route_coords[2] = z_coord

        if np.any(np.isnan(route_coords)):
            route_coords = cur_pos

        dir_vec_2d = route_coords[:2] - cur_pos[:2]
        if route_coords[2] > -2:
            route_coords[2] = 2

    #     print(f"next route point: {route_coords}")
    # print(f"waypoint prediction time:{time.time()-time1}")

    time1 = time.time()
    # low level path
    rel_trans = route_coords - cur_pos
    yaw = np.arctan2(rel_trans[1], rel_trans[0])
    new_quat = R.from_euler('z', yaw, degrees=False).as_quat()
    # new_quat = R.from_quat(rot_quat) * R.from_quat(cur_ori)
    # new_quat = list(new_quat.as_quat())
    new_pos = route_coords
    new_pose = convert_airsim_pose(list(new_pos)+list(new_quat))

    # calculate step size
    dist = np.abs(rel_trans)
    step_size = np.abs(np.rad2deg(yaw)) // 15 + dist[2] // 2 + np.sqrt(dist[0]**2+dist[1]**2) // 5

    print(f"low level planning time: {time.time()-time1}")
    print(f"curr pose: {curr_pose}, new pose: {new_pose}, object point: {route_coords}")
    return int(step_size), new_pose


def explore_pipeline_by_sam(
        curr_pose,
        llm, vlm,
        image_path: List[str],
        rgb_imgs: List[np.ndarray],
        dep_imgs: List[np.ndarray],
        obs_poses: List[np.ndarray],
        navigation_instruction: str,
        scene_objects: List[str], landmarks_route: List[str]
):
    # image caption
    time1 = time.time()
    caption_prompt = landmark_caption_prompt_builder(scene_objects)
    print(f"caption prompt: {caption_prompt}")

    total_observed_obj = []
    for img_p in image_path:
        try:
            caption_res_str = llm.query_api(caption_prompt, image_path=img_p, show_response=False)
            _, unique_obj = reformat_dino_prompt(caption_res_str)
            total_observed_obj += unique_obj
        except Exception as e:
            print(e)
    total_observed_obj = list(set(total_observed_obj))
    obs_obj_str = ".".join(total_observed_obj)

    # print(f"obs_str:{obs_obj_str}")
    # print(f"scene caption time: {time.time()-time1}")
    # # obs_obj_str = 'billboard.antenna next to the bulletin.road. woods. wire pole'

    # prompting next waypoint
    time1 = time.time()
    route_predict_prompt = route_planning_prompt_builder(obs_obj_str, navigation_instruction, landmarks_route[0])
    print(f"route planning prompt: {route_predict_prompt}")
    route_predicted = llm.query_api(route_predict_prompt, show_response=False)
    route_predicted, _ = reformat_dino_prompt(route_predicted)
    # # route_predicted = "wire pole.billboard.street lamp.road"
    # print(f"query time: {time.time()-time1}")
    # print("route_predict_prompt: ", route_predict_prompt)
    # # print("obs_obj_str: ", obs_obj_str)
    # print("route_predicted: ", route_predicted)

    # build semantic point cloud
    semantic_pc = []
    seg_succ_all = False
    for j in range(len(rgb_imgs)):
        rgb_img = rgb_imgs[j]
        dep_img = dep_imgs[j].squeeze()
        pose = obs_poses[j]

        route_mask, seg_succ = vlm.greedy_mask_predict(rgb_img, route_predicted, visualize=False)
        seg_succ_all = seg_succ_all or seg_succ
        if seg_succ:
            part_pc, filter_idx = convert_global_pc(dep_img, 90, pose, route_mask)
            semantic_part_pc = part_pc[filter_idx]
            # semantic_part_pc = statistical_filter(semantic_part_pc)
            if len(semantic_part_pc > 0):
                semantic_pc.append(semantic_part_pc)

    if len(semantic_pc) > 0:
        semantic_pc = np.concatenate(semantic_pc, axis=0)
        if len(semantic_pc) > 30:
            semantic_pc, _ = statistical_filter(semantic_pc, k=30)
        else:
            semantic_pc = np.zeros((1, 3))
    else:
        semantic_pc = np.zeros((1, 3))

    # visualize_point_cloud(semantic_pc)

    # route point prediction
    cur_pos = np.array([curr_pose.position.x_val, curr_pose.position.y_val, curr_pose.position.z_val])
    cur_ori = np.array([curr_pose.orientation.x_val, curr_pose.orientation.y_val, curr_pose.orientation.z_val, curr_pose.orientation.w_val])
    cur_pose= np.concatenate([cur_pos, cur_ori], axis=0)

    if not seg_succ_all:
        route_coords = cur_pos
    else:
        route_coords = np.mean(semantic_pc, axis=0)
        if not np.any(route_coords):        # if all zeros
            route_coords = cur_pos
        alpha = 0.6
        route_coords = alpha * route_coords + (1-alpha) * cur_pos
        if np.any(np.isnan(route_coords)):
            route_coords = cur_pos

        dir_vec_2d = route_coords[:2] - cur_pos[:2]
        if route_coords[2] > -2:
            route_coords[2] = 2

    #     print(f"next route point: {route_coords}")
    #
    # print(f"waypoint prediction time:{time.time()-time1}")

    time1 = time.time()
    # low level path
    rel_trans = route_coords - cur_pos
    yaw = np.arctan2(rel_trans[1], rel_trans[0])
    new_quat = R.from_euler('z', yaw, degrees=False).as_quat()
    # new_quat = R.from_quat(rot_quat) * R.from_quat(cur_ori)
    # new_quat = list(new_quat.as_quat())
    new_pos = route_coords
    new_pose = convert_airsim_pose(list(new_pos)+list(new_quat))

    # calculate step size
    dist = np.abs(rel_trans)
    step_size = np.abs(np.rad2deg(yaw)) // 15 + dist[2] // 2 + np.sqrt(dist[0]**2+dist[1]**2) // 5

    # print(f"low level planning time: {time.time()-time1}")

    return int(step_size), new_pose


def CityNav(scene_id, max_step_size=200, cfg=None):
    # load data
    data_loader = AirVLNDataLoader("files/trajectory", scene_ids=[scene_id])
    navi_tasks_grouped = data_loader.full_navi_data_grouped
    navi_scene_infos = data_loader.scene_info

    # load LLM
    llm = OpenAI_LLM(
        max_tokens=4096,
        model_name="gpt-4o",
        api_key="OPENAI_API_KEYS",
        cache_name="navigation",
        finish_reasons=["stop", "length"],
    )
    vlm = load_model(
        "Grounded_Sam_Lite/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "/home/vincent/py-pro/AirVLN-main/AirVLN/weights/groundingdino_swint_ogc.pth"
    )

    # navigation
    for scene_id, navi_tasks in navi_tasks_grouped.items():
        # load scene info
        scene_info = navi_scene_infos[scene_id]
        scene_objects = scene_info["scene_objects"]

        # load graph
        mem_graph = NavigationGraph(
            f"/home/vincent/py-pro/AirVLN-main/AirVLN/files/gt/mem_env_{scene_id}/graph_merged.pkl")

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
        sr_cnt = 0.0
        spl_cnt = 0.0
        total_cnt = len(navi_tasks)
        ne = []
        res_strs = []
        nav_evaluator = CityNavEvaluator()


        for t_idx, navi_task in enumerate(navi_tasks):
            # if t_idx != 0:
            #     continue

            frames = []

            landmarks = navi_task["reference_landmark_route"]
            reference_graph_idx_route = navi_task["reference_graph_idx_route"]
            instruction = navi_task["instruction"]
            gt_path_length = navi_task["path_length"]

            start_graph_node = reference_graph_idx_route[0]
            second_graph_node = reference_graph_idx_route[1]

            start_pos = mem_graph.get_node_data(start_graph_node)["position"]
            second_pos = mem_graph.get_node_data(second_graph_node)["position"]

            step_size = 0
            hist_step_size = []
            curr_pose = convert_airsim_pose(navi_task["start_pose"])
            # curr_pose = convert_airsim_pose([55.87352793, -30.30470698, -21.65148405, 0.0, 0.0, 0., 1])
            target_pose = convert_airsim_pose(navi_task["end_pose"])

            # set env
            tool.setPoses([[curr_pose]])

            # todo: set max step size=60
            total_mv_length = 0.0
            while step_size < max_step_size:
                print("step size: {}".format(step_size))
                print(f"pose: {list(curr_pose.position)+list(curr_pose.orientation)}")
                time_s = time.time()
                # get observation
                try:
                    pano_obs, pano_pose = get_pano_observations(curr_pose, tool, scene_id=scene_id)
                    pano_obs_imgs = [pano_obs[6][0], pano_obs[7][0], pano_obs[0][0], pano_obs[1][0], pano_obs[2][0], pano_obs[4][0]]
                    pano_obs_deps = [pano_obs[6][1], pano_obs[7][1], pano_obs[0][1], pano_obs[1][1], pano_obs[2][1], pano_obs[4][1]]

                    pano_obs_imgs_path = ["obs_imgs/rgb_obs_{}.png".format(view_drc.replace(" ", "_")) for view_drc in
                                          ObservationDirections+["back"]]
                    pano_obs_deps_path = ["obs_imgs/dep_obs_{}.npy".format(view_drc.replace(" ", "_")) for view_drc in
                                          ObservationDirections+["back"]]
                    pano_pose_path = ["obs_imgs/pose_{}.npy".format(view_drc.replace(" ", "_")) for view_drc in
                                          ObservationDirections+["back"]]
                    # print(pano_obs_deps[0])

                    for j in range(len(pano_obs_imgs_path)):
                        cv2.imwrite(pano_obs_imgs_path[j], pano_obs_imgs[j])
                        np.save(pano_obs_deps_path[j], pano_obs_deps[j])
                        np.save(pano_pose_path[j], pano_pose[j])

                        pano_obs_depvis = (pano_obs_deps[j].squeeze() * 255).astype(np.uint8)
                        pano_obs_depvis = np.stack([pano_obs_depvis for _ in range(3)], axis=2)

                        cv2.imwrite(pano_obs_deps_path[j].replace("npy", "png"), pano_obs_depvis)

                    # visualize
                    cv2.imwrite(
                        f"/home/vincent/py-pro/AirVLN-main/AirVLN/files/videos/res_env_{scene_id}/temp/front_{step_size}.png", pano_obs[0][0])

                except Exception as e:
                    data_dict['pred_traj'].append(list(curr_pose.position))
                    print(f"task idx: {t_idx}. Step size: {step_size}. success: False, failed to get images. Exception: {e}")
                    break
                print(f"observation time: {time.time()-time_s}")

                # calculate current position to the graph
                dist2graph = np.linalg.norm(np.array(list(curr_pose.position))-second_pos)

                # explore or exploit
                # exploit
                if dist2graph < 20:
                    print("find the memory graph node!!!")
                    with open(pano_obs_imgs_path[0], "rb") as file:
                        imgf = file.read()
                    with open(pano_obs_imgs_path[-1], "rb") as file:
                        imgb = file.read()

                    obs = {
                        "pos": np.array(list(curr_pose.position)),
                        "image": [imgf, imgb]
                    }
                    new_node = mem_graph.add_vertix(obs)
                    mem_graph.add_edge(new_node, second_graph_node)

                    result = pipeline.full_pipeline(mem_graph, start_node=new_node, landmarks=landmarks[1:])

                    # evaluate
                    walk = [a[0] for a in result["walk"]]
                    rest_steps = int(min(100-step_size, len(walk)))
                    for k in range(1, rest_steps):
                        p1_idx = walk[k]
                        p2_idx = walk[k-1]
                        step_mv_length = np.linalg.norm(mem_graph._pos[p1_idx] - mem_graph._pos[p2_idx])
                        total_mv_length += step_mv_length

                    # for k in range(len(walk)):
                    #     cur_pos_idx = walk[k]
                    #     cur_pos = mem_graph._pos[cur_pos_idx]
                    #     tar_pos = np.array(list(target_pose.position))
                    #     mid_ne = np.linalg.norm((cur_pos - tar_pos))
                    #     print(f"{k} {mid_ne}")

                    # visualize
                    for k in range(rest_steps):
                        pano = []
                        for t in mem_graph._images[walk[k]]:
                            img = np.array(PIL.Image.open(io.BytesIO(t)))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            pano.append(img)
                        cv2.imwrite(f"/home/vincent/py-pro/AirVLN-main/AirVLN/files/videos/res_env_{scene_id}/temp/front_{step_size+k}.png", pano[0])
                        cv2.imwrite(f"/home/vincent/py-pro/AirVLN-main/AirVLN/files/videos/res_env_{scene_id}/temp/back_{step_size+k}.png", pano[1])

                    stop_pos = mem_graph.get_node_data(walk[rest_steps - 1])["position"]
                    new_pose = convert_airsim_pose(list(stop_pos)+list(curr_pose.orientation))
                    tool.setPoses([[new_pose]])
                    curr_pose = new_pose
                    step_size += rest_steps

                    break
                # explore
                else:
                    time1 = time.time()
                    sz, new_pose = explore_pipeline_by_dino(curr_pose, llm, vlm, pano_obs_imgs_path[:5], pano_obs_imgs[:5], pano_obs_deps[:5], instruction, scene_objects, landmarks)
                    print(f"explore pipeline time: {time.time()-time1}")
                    step_mv_length = np.linalg.norm(np.array(list(curr_pose.position)) - np.array(list(new_pose.position)))
                    total_mv_length += step_mv_length

                    tool.setPoses([[new_pose]])
                    curr_pose = new_pose

                    step_size += sz
                    hist_step_size.append(sz)

                    # # visualize
                    # obs_responses = tool.getImageResponses(camera_id='front_0')
                    # img_rgb = obs_responses[0][0][0]
                    # img_dep = obs_responses[0][0][1]
                    # cv2.imwrite(
                    #     f"/home/vincent/py-pro/AirVLN-main/AirVLN/files/videos/res_env_1/temp/front_{step_size}.png", img_rgb)

                    print(f"total reference time: {time.time() - time_s}")
                    print(hist_step_size)
                    if len(hist_step_size)>=4 and sum(hist_step_size[-4:-1]) == 0.0:
                        print(f"task idx: {t_idx}. total_steps: {step_size}. success: False. Stuck!!")
                        break

            stop_pos = np.array(list(curr_pose.position))
            target_pos = np.array(list(target_pose.position))

            dist = np.linalg.norm((stop_pos - target_pos))

            if dist < 20:
                sr_cnt += 1
                t_spl_cnt = gt_path_length / max(gt_path_length, total_mv_length)
                spl_cnt += t_spl_cnt
                res_str = f"task idx: {t_idx}. success: True, spl_cnt:{t_spl_cnt}, total_steps: {step_size}. distance error: {dist}."

                print(res_str)
            else:
                res_str = f"task idx: {t_idx}. success: False, spl_cnt: 0, total_steps: {step_size}. distance error: {dist}."
                print(res_str)
            ne.append(dist)
            res_strs.append(res_str)

        for res_str in res_strs:
            print(res_str)
        print(f"SR: {sr_cnt/total_cnt}, NE: {sum(ne)/total_cnt}, SPL: {spl_cnt/total_cnt}")


##### test on airvln-s ####################
def CityNav_v2(scene_id, split, max_step_size=200, vlm_name="dino", vis=False):
    data_root = os.path.join(DATA_ROOT, f"gt_by_env/{env_id}/{split}_nms.json")
    # graph_root = os.path.join(DATA_ROOT, f"lm_graphs_circle_nms_v2_wo_start/{env_id}/{split}")
    graph_root = os.path.join(DATA_ROOT, f"lm_graphs_circle/{env_id}/{split}")

    # change the graph root for ablation
    graph_root = os.path.join(DATA_ROOT, f"lm_graphs_0.7_nms_wo_start/{env_id}/{split}")

    with open(data_root, 'r') as f:
        navi_tasks = json.load(f)['episodes']

    nav_evaluator = CityNavEvaluator()

    # load LLM
    llm = OpenAI_LLM_v3(
        max_tokens=4096,
        model_name="gpt-4o",
        api_key="OPENAI_API_KEY",
        client_type="openai",
        cache_name="navigation",
        finish_reasons=["stop", "length"],
    )

    if vlm_name == "dino":
        vlm = load_model(
            "Grounded_Sam_Lite/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "Grounded_Sam_Lite/weights/groundingdino_swint_ogc.pth"
        )
    elif vlm_name == "sam":
        vlm = GroundedSam(
            dino_checkpoint_path="Grounded_Sam_Lite/weights/groundingdino_swint_ogc.pth",
            sam_checkpoint_path="Grounded_Sam_Lite/weights/sam_vit_h_4b8939.pth"
        )

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

    # navigation pipeline
    for i in tqdm(range(len(navi_tasks))):
        navi_task = navi_tasks[i]
        # load scene info
        episode_id = navi_task['episode_id']
        # if episode_id not in selet_episodes:
        #     continue

        print(f"================================ start episode {episode_id} ==================================")
        # load graph
        mem_graph = NavigationGraph(os.path.join(graph_root, f"{episode_id}.pkl"))

        landmarks = navi_task["instruction"]["landmarks"]
        if len(landmarks) == 0:
            continue

        object_info = []
        instruction = navi_task["instruction"]['instruction_text']
        reference_path = navi_task['reference_path']
        start_pos = reference_path[0][:3]
        end_pos = reference_path[-1][:3]

        data_dict = {
            "episode_id": episode_id,
            "gt_traj": [pose[:3] for pose in reference_path],
            "pred_traj": []
        }

        step_size = 0
        hist_step_size = []

        curr_pose = convert_airsim_pose(navi_task["start_position"]+navi_task["start_rotation"])
        target_pose = convert_airsim_pose(navi_task["goals"][0]['position']+[0, 0, 0, 1])

        # set env
        tool.setPoses([[curr_pose]])

        while step_size < max_step_size:
            # print("step size: {}".format(step_size))
            # print(f"pose: {list(curr_pose.position)+list(curr_pose.orientation)}")
            time_s = time.time()
            # get observation
            try:
                pano_obs, pano_pose = get_pano_observations(curr_pose, tool, scene_id=scene_id)
                pano_obs_imgs = [pano_obs[6][0], pano_obs[7][0], pano_obs[0][0], pano_obs[1][0], pano_obs[2][0], pano_obs[4][0]]
                pano_obs_deps = [pano_obs[6][1], pano_obs[7][1], pano_obs[0][1], pano_obs[1][1], pano_obs[2][1], pano_obs[4][1]]
                pano_obs_poses = [pano_pose[6], pano_pose[7], pano_pose[0], pano_pose[1], pano_pose[2], pano_pose[4]]

                pano_obs_imgs_path = ["obs_imgs/rgb_obs_{}.png".format(view_drc.replace(" ", "_")) for view_drc in
                                      ObservationDirections+["back"]]
                pano_obs_deps_path = ["obs_imgs/dep_obs_{}.npy".format(view_drc.replace(" ", "_")) for view_drc in
                                      ObservationDirections+["back"]]
                pano_pose_path = ["obs_imgs/pose_{}.npy".format(view_drc.replace(" ", "_")) for view_drc in
                                      ObservationDirections+["back"]]

                for j in range(len(pano_obs_imgs_path)):
                    cv2.imwrite(pano_obs_imgs_path[j], pano_obs_imgs[j])
                    np.save(pano_obs_deps_path[j], pano_obs_deps[j])
                    np.save(pano_pose_path[j], pano_obs_poses[j])

                    pano_obs_depvis = (pano_obs_deps[j].squeeze() * 255).astype(np.uint8)
                    pano_obs_depvis = np.stack([pano_obs_depvis for _ in range(3)], axis=2)

                    cv2.imwrite(pano_obs_deps_path[j].replace("npy", "png"), pano_obs_depvis)

                # visualize
                if vis:
                    cv2.imwrite(
                        f"/home/vincent/py-pro/AirVLN-main/AirVLN/files/videos/res_env_{scene_id}/temp/front_{step_size}.png", pano_obs[0][0])

            except Exception as e:
                data_dict['pred_traj'].append(list(curr_pose.position))
                print(f"task idx: {i}. Step size: {step_size}. success: False, failed to get images. Exception: {e}")
                break
            # print(f"observation time: {time.time()-time_s}")

            # calculate current position to the graph
            cls_node = find_closest_node(mem_graph._graph, list(curr_pose.position), thresh=20)
            print(f"closest node: {cls_node}")

            # explore or exploit
            # exploit
            if cls_node is not None:
                print("find the memory graph node!!!")
                with open(pano_obs_imgs_path[0], "rb") as file:
                    imgf = file.read()
                with open(pano_obs_imgs_path[-1], "rb") as file:
                    imgb = file.read()

                obs = {
                    "pos": np.array(list(curr_pose.position)),
                    "image": [imgf, imgb]
                }
                new_node = mem_graph.add_vertix(obs)
                mem_graph.add_edge(new_node, cls_node)

                result = pipeline.full_pipeline(mem_graph, start_node=new_node, landmarks=landmarks, alpha=0.0001)

                # evaluate
                walk = [a[0] for a in result["walk"]]

                rest_steps = int(min(max_step_size-step_size, len(walk)))
                rest_walks = walk[:rest_steps]

                # stop_pos = mem_graph.get_node_data(rest_walks[-1])["position"]
                # ne = np.linalg.norm(np.array(end_pos) - np.array(stop_pos))

                data_dict['pred_traj'].extend([mem_graph.get_node_data(node)["position"] for node in rest_walks])

                stop_pos = mem_graph.get_node_data(rest_walks[-1])["position"]
                curr_pose = convert_airsim_pose(list(stop_pos) + list(curr_pose.orientation))
                tool.setPoses([[curr_pose]])

                step_size += rest_steps

                # visualize
                if vis:
                    for k in range(rest_steps):
                        pano = []
                        for t in mem_graph._images[walk[k]]:
                            img = np.array(PIL.Image.open(io.BytesIO(t)))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            pano.append(img)
                        cv2.imwrite(f"/home/vincent/py-pro/AirVLN-main/AirVLN/files/videos/res_env_{scene_id}/temp/front_{step_size+k}.png", pano[0])
                        cv2.imwrite(f"/home/vincent/py-pro/AirVLN-main/AirVLN/files/videos/res_env_{scene_id}/temp/back_{step_size+k}.png", pano[1])

                break
            # explore
            else:
                print("no memory graph reached, keep exploring ...")
                time1 = time.time()

                if vlm_name == "dino":
                    _, new_pose = explore_pipeline_by_dino(
                        curr_pose, llm, vlm,
                        pano_obs_imgs_path[:5],
                        pano_obs_imgs[:5],
                        pano_obs_deps[:5],
                        instruction, object_info, landmarks)
                elif vlm_name == "sam":
                    _, new_pose = explore_pipeline_by_sam(
                        curr_pose, llm, vlm,
                        pano_obs_imgs_path[:5],
                        pano_obs_imgs[:5],
                        pano_obs_deps[:5],
                        pano_obs_poses[:5],
                        instruction, object_info, landmarks)

                print(f"explore pipeline time: {time.time()-time1}")

                sz, mid_coords = calculate_movement_steps(list(curr_pose.position), list(new_pose.position), step_size=5)
                data_dict['pred_traj'].extend(mid_coords)
                print(mid_coords)
                tool.setPoses([[new_pose]])
                curr_pose = new_pose

                step_size += sz
                hist_step_size.append(sz)

                # visualize
                if vis:
                    obs_responses = tool.getImageResponses(camera_id='front_0')
                    img_rgb = obs_responses[0][0][0]
                    img_dep = obs_responses[0][0][1]
                    cv2.imwrite(f"/home/vincent/py-pro/AirVLN-main/AirVLN/files/videos/res_env_1/temp/front_{step_size}.png", img_rgb)

                print(f"total reference time: {time.time() - time_s}")
                print(hist_step_size)
                if len(hist_step_size)>=4 and sum(hist_step_size[-4:-1]) == 0.0:
                    print(f"task idx: {i}. total_steps: {step_size}. success: False. Stuck!!")
                    break

        stop_pos = np.array(list(curr_pose.position))
        target_pos = np.array(list(target_pose.position))
        ne = np.linalg.norm(np.array(target_pos) - np.array(stop_pos))

        if ne < 20:
            print(f"############## episode {episode_id} success, ne: {ne}, step size: {step_size}")
        else:
            print(f"############## episode {episode_id} failed, ne: {ne}")

        nav_evaluator.update(data_dict)

    nav_evaluator.log_metrics()


################ generate random walk traj for validation unseen ###############3
def generate_random_walk(env_id, split):
    data_prefix = "/media/vincent/Seagate Expansion Drive/airvln/files"
    data_root = os.path.join(data_prefix, f"gt_by_env/{env_id}/{split}_lm.json")
    save_root = os.path.join(data_prefix, f"gt_by_env/{env_id}/{split}_lm_randwalk.json")

    with open(data_root, 'r') as f:
        navi_tasks = json.load(f)['episodes']

    # load env
    machines_info_xxx = [
        {
            'MACHINE_IP': '127.0.0.1',
            'SOCKET_PORT': 30000,
            'MAX_SCENE_NUM': 8,
            'open_scenes': [env_id],
        },
    ]

    tool = AirVLNSimulatorClientTool(machines_info=machines_info_xxx)
    tool.run_call()

    navi_tasks_randwalk = []
    ori_node_count = 0.0
    after_node_count = 0.0
    for i in tqdm(range(len(navi_tasks))):
        navi_task = navi_tasks[i]

        episode_id = navi_task['episode_id']
        new_episode_id = episode_id + '-rand'
        reference_path = navi_task['reference_path']
        reference_action = navi_task['actions']

        ori_node_count += len(reference_path)
        print(f"reference path: {len(reference_path)}, reference_action:{len(reference_action)}")

        start_pose = reference_path[3]
        curr_pose = convert_airsim_pose(start_pose[:3]+list(airsim.to_quaternion(*start_pose[3:])))

        # set env
        tool.setPoses([[curr_pose]])

        step_size = 0
        hist_act_codes = []
        hist_pose = []
        while step_size < len(reference_path):
            act_code = np.random.randint(1, 6)
            hist_act_codes.append(act_code)
            ori = curr_pose.orientation

            # print(list(ori))
            ori_eu = airsim.to_eularian_angles(ori)
            ori_quat = airsim.to_quaternion(*ori_eu)
            # print(list(ori_quat))

            hist_pose.append(list(curr_pose.position)+list(ori_eu))

            if act_code == 0:
                break

            new_pose = getPoseAfterMakeActions(curr_pose, [act_code])
            new_pos = list(new_pose.position)
            if new_pos[2] > -2:
                new_pos[2] = -2

            tool.setPoses([[new_pose]])
            step_size += 1
            curr_pose = new_pose

        print(f"hist path: {len(hist_pose)}, hist action: {len(hist_act_codes)}")
        rand_navi_task = navi_task.copy()
        rand_navi_task['episode_id'] = new_episode_id
        rand_navi_task['reference_path'] = hist_pose
        rand_navi_task['actions'] = hist_act_codes

        navi_tasks_randwalk.append(rand_navi_task)

        after_node_count += len(hist_pose)

    print(f"before node count : {ori_node_count}, after node count: {ori_node_count+after_node_count}")
    new_navi_tasks = navi_tasks + navi_tasks_randwalk
    with open(save_root, 'w') as f:
        json.dump({"episodes": new_navi_tasks}, f, indent=4)


if __name__ == '__main__':
    env_id = 2
    split = "val_seen"

    CityNav_v2(env_id, split, max_step_size=60, vis=False)


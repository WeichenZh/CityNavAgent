import os
import json
import sys
import uuid
sys.path.append("../")
from lm_nav.navigation_graph import NavigationGraph


class AirVLNDataLoader:
    def __init__(self, root, scene_ids=[]):
        self.root = root
        self.scene_ids = scene_ids if len(scene_ids) > 0 else list(range(1, 26))
        self.full_navi_data, self.full_navi_data_grouped, self.scene_info = self._load_full_data()

    def _load_full_data(self):
        full_navi_data = []
        full_navi_data_grouped = {}
        scene_info = {}
        for scene_id in self.scene_ids:
            data_path = os.path.join(self.root, f"env_{scene_id}_data_w_traj.json")

            with open(data_path, 'r') as f:
                navi_data = json.load(f)

            navi_landmarks = navi_data["landmarks"]
            scene_objects = navi_data["scene_objects"]

            navi_tasks = navi_data["navigation_tasks"]
            full_navi_data += navi_tasks
            full_navi_data_grouped[scene_id] = navi_tasks
            scene_info[scene_id] = {
                "landmarks": navi_landmarks,
                "scene_objects": scene_objects
            }

        return full_navi_data, full_navi_data_grouped, scene_info


def gen_gt_data(scene_id):
    root = "files/trajectory"

    graph = NavigationGraph(f"files/gt/mem_env_{scene_id}/graph_merged.pkl")

    target_file = os.path.join(root, f"env_{scene_id}_data.json")
    source_file = os.path.join(root, f"env_{scene_id}_landmark_notes.json")

    with open(source_file, 'r') as f:
        source_data = json.load(f)

    target_data = {}
    landmarks = source_data["landmarks"]
    target_data["landmarks"] = landmarks
    target_data["scene_objects"] = source_data["scene_objects"]
    target_data["navigation_tasks"] = []
    for task in source_data["navigation_tasks"]:
        s_uuid = str(uuid.uuid4())
        s_uuid = "".join(s_uuid.split("-"))

        reference_idx_route = task["landmark_idx_route"]
        degree = task["degree"]
        instruction = task["instruction"]

        start_route_idx = reference_idx_route[0]
        end_route_idx   = reference_idx_route[-1]
        start_pose = landmarks[start_route_idx-1][2]
        end_pose   = landmarks[end_route_idx-1][2]

        reference_route = []
        reference_graph_idx_route = []
        for idx in reference_idx_route:
            reference_route.append(landmarks[idx-1][1])
            dist, node_idx = graph.find_closest_node(np.array(landmarks[idx-1][2][:3]))
            reference_graph_idx_route.append(node_idx)

        path_dist = 0.0
        for j in range(1, len(reference_idx_route)):
            step_dist = np.linalg.norm(graph._pos[reference_graph_idx_route[j]] - graph._pos[reference_graph_idx_route[j-1]])
            path_dist += step_dist
            # print(step_dist)

        new_task = {
            "task_id": s_uuid,
            "reference_landmark_route": reference_route,
            "reference_landmark_idx_route": reference_idx_route,
            "reference_graph_idx_route": reference_graph_idx_route,
            "instruction": instruction,
            "start_pose": start_pose,
            "end_pose": end_pose,
            "path_length": path_dist,
            "degree": degree,
            "scene_id": scene_id
        }

        target_data["navigation_tasks"].append(new_task)

    with open(target_file, 'w') as f:
        json.dump(target_data, f)


def dataset_metric():
    # load data
    data_loader = AirVLNDataLoader("files/trajectory", scene_ids=[1, 5, 7, 11, 14, 16])
    navi_tasks_grouped = data_loader.full_navi_data_grouped
    navi_scene_infos = data_loader.scene_info

    total_gt_path_len = {"easy": [], "middle": [], "hard": []}
    for scene_id, navi_tasks in navi_tasks_grouped.items():
        # load scene info
        scene_info = navi_scene_infos[scene_id]
        scene_objects = scene_info["scene_objects"]

        for t_idx, navi_task in enumerate(navi_tasks):

            frames = []

            landmarks = navi_task["reference_landmark_route"]
            reference_graph_idx_route = navi_task["reference_graph_idx_route"]
            instruction = navi_task["instruction"]
            gt_path_length = navi_task["path_length"]
            degree = navi_task["degree"]

            total_gt_path_len[degree].append(gt_path_length)

            start_graph_node = reference_graph_idx_route[0]
            second_graph_node = reference_graph_idx_route[1]

    print(total_gt_path_len["easy"])
    print(total_gt_path_len["middle"])
    print(total_gt_path_len["hard"])
    print(sum(total_gt_path_len['easy']) / len(total_gt_path_len['easy']))
    print(sum(total_gt_path_len['middle']) / len(total_gt_path_len['middle']))
    print(sum(total_gt_path_len['hard']) / len(total_gt_path_len['hard']))

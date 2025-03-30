import re
import sys
sys.path.append("../../")
from airsim_plugin.airsim_settings import DefaultAirsimActionNames, DefaultAirsimActionCodes, ObservationDirections

action_space = "Action Space:\nforward (go straight), left (rotate left), right (rotate right), stop (end navigation)\n\n"
prompt_template = 'Navigation Instructions:\n"{}"\nAction Sequence:\n'


def reformat_dino_prompt(prompt):
    unique_obj = set()
    prompt_obj = prompt.lower().split(".")
    for o in prompt_obj:
        if o.strip(" ") not in unique_obj:
            unique_obj.add(o.strip(" "))
    formatted_prompt = ".".join(list(unique_obj))
    return formatted_prompt, list(unique_obj)

def build_prompt(instructions):
    prompt = action_space
    # prompt += prompt_template.format(instructions, action_space, "1. forward\n2.")
    prompt += prompt_template.format(instructions, action_space)
    return prompt


def get_navigation_lines(nav, env, landmarks, traffic_flow, step_id=0):
    actions = nav.actions
    states = nav.states

    assert len(actions) == len(states)

    lines = list()
    is_action = list()
    while step_id < len(actions):
        action = actions[step_id]

        # print step number and action
        line = f"{step_id}. {action}"
        if action != "init":
            lines.append(line)
            is_action.append(True)

        # print current env observations
        observations = env.get_observations(states, step_id, landmarks, traffic_flow)
        observations_str = get_observations_str(observations)

        if observations_str:
            line = observations_str
            lines.append(line)
            is_action.append(False)

        step_id += 1

    # print number of input step if sequence not finished
    if actions[-1] != "stop":
        line = f"{len(actions)}."
        lines.append(line)
        is_action.append(False)

    assert len(lines) == len(is_action)
    return lines, is_action


def get_observations_str(observations):
    observations_strs = list()

    if "traffic_flow" in observations:
        traffic_flow = observations["traffic_flow"]
        observations_strs.append(
            f"You are aligned {traffic_flow} the flow of the traffic."
        )

    if "intersection" in observations:
        num_ways = observations["intersection"]
        observations_strs.append(f"There is a {num_ways}-way intersection.")

    if "landmarks" in observations:
        directions = [
            "on your left",
            "slightly left",
            "ahead",
            "slightly right",
            "on your right",
        ]
        for direction, landmarks in zip(directions, observations["landmarks"]):
            if len(landmarks) > 0:
                landmarks = " and ".join(landmarks)
                landmarks = landmarks[0].upper() + landmarks[1:]
                observations_strs.append(f"{landmarks} {direction}.")

    return " ".join(observations_strs)


def landmarks_extraction_prompt_builder(target, navi_hint):
    prompt_str = (
            (
                "Please specific all landmarks in the following navigation hint that helps to navigate to the target. "
                "You should extract the name of landmark and its characteristics based on navigation hint. You should NOT output other information. \n"
            )
            + "target: {}\n".format(target)
            + "Navigation hint: {}".format(navi_hint)
            + "Your output format should be: <landmark name>: <landmark characteristics>|<landmark name>: <landmark characteristics>|...\n"
            + "You should focus more on landmarks' color and shape. \n"
            # + "\nYour output should be in following format: \n"
            # + "[{observed object}: {<color>, <texture>, <height>, <width> and <other characteristics>};...]\n"

            # + "landmark description should be one or two sentences.\n"
            + """
        Here are some output examples:

        twin buildings: tall and silver | a big tree: green | cocacola store: red sign

        """
    )
    # "landmark description should be no more 6 short words or phases, seperated by a comma."

    return prompt_str


def visual_observation_prompt_builder():
    # landmark_str = ""
    # for landmark in landmarks:
    #     landmark_str += "{}: {}\n".format(landmark, landmarks[landmark])

    prompt_str = (
            "Given an image, please describe the object and scenes with its characteristics. \n"
            # + "The characteristics must include its color, texture, height and width. \n"
            # + "Besides, you need to tell whether you observed landmarks provided below: \n"
            # + "{landmark_str}".format(landmark_str=landmark_str)
            + "\nYour output should be in following format: \n"
            + "[{observed object}: {characteristics};[{observed object}: {characteristics}...]\n"
            # + '{"observed landmark": [{landmark1}, {landmark2}, {landmark3}...]}'
            + """
        Here are some output examples: 

        [a skycraper: silver, smooth, tall and slim; cloudy sky: white, fulfilled, high and wide; garden: green and red, messy, short and small]
        """
    )

    return prompt_str


def open_ended_action_manager_prompt_builder(
        landmark_prompt,
        current_text,
        experience,
        relate_knowledge,
        fire=False,
        exclude_actions=None,
):
    experience_str = ""
    if experience != []:
        for i, exp in enumerate(experience):
            experience_str += f"{i + 1}.\n"
            for idx in range(len(exp[0])):
                experience_str += f"state: {exp[0][idx]}\naction: {exp[1][idx]}\n"
    knowledge_str = ""
    if relate_knowledge != None:
        for i, k in enumerate(relate_knowledge):
            knowledge_str += f"{i + 1}. {k[0]} appeared {k[1]} times.\n"
    sudden_message = ""

    if fire:
        sudden_message = "Usually, fire will happen in the place with more buildings. It will cause much smoke.\n"

    all_action_list = [
        "MOVE FORWARD",
        "MOVE FRONT_LEFT",
        "MOVE LEFT",
        "MOVE BACK_LEFT",
        "MOVE BACK",
        "MOVE BACK_RIGHT",
        "MOVE RIGHT",
        "MOVE FRONT_RIGHT",
        "MOVE ASCEND",
        "MOVE DESCEND",
        "STOP",
    ]
    action_list = []
    for i in range(len(all_action_list)):
        if exclude_actions is not None and i in exclude_actions:
            continue
        action_list.append(all_action_list[i])

    prompt_str = f"""
    You are an AI agent helping to control the drone to navigate the street view. 
    {landmark_prompt}
    Please provide the next action you should take. You can ONLY use the following actions: {action_list}. Also, you need to provide the reason why you take the action.

    Example output: 
    MOVE FORWARD: the building in front appeared for a lot of times with the target, I think it's helpful for finding the target.
    MOVE LEFT: the target building is on the left
    MOVE FORWARD: this place is arrived in earlier experience, so you should follow the earlier path.
    STOP: you have arrived at the target.
    MOVE ASCEND: there is nothing helpful for the target, let's ascent for better view.
    MOVE FORWARD: the building in front appeared for a lot of times with the target, I think it's helpful for finding the target.

    Here are some useful information that may help you to make the decision:

    You should notice the effect of weather. For example, snowy weather makes the street view filled with white, foggy weather makes your sight shorter. 

    {sudden_message}

    In your previous experience, you have observed the following experience, 1. means the index of experience, state means your view of the environment, action means the action you take after observing the state, and the next state means the view of the environment after taking the action. 
    [[
{experience_str}
    ]]
    Based on the previous experience, you have the following knowledge. They are the number of times each object appeared together with the target. If you have no idea about the target, you can also try to find these objects.
    [[
{knowledge_str}
    ]]

    Your current environment observation is: {current_text}
    """

    return prompt_str

def cot_prompt_builder_p1(
        navigation_instruction,
        prev_actions,
):
    prompt_str = f"""
    You are an AI agent helping to control the drone to finish the navigation task.
    Your navigation task is {navigation_instruction}.
    Your previous action sequence is {prev_actions}.
    Based on your navigation task and previous action, what's your current sub-goal to reach.
    """
    return prompt_str


def cot_prompt_builder_p2(
        navigation_instruction,
        prev_actions,
        current_subgoal,
        current_observation,
        current_position,
):
    prompt_str = f"""
    You are an AI agent helping to control the drone to finish the navigation task.
    Your current sub-goal to reach is {current_subgoal}.
    Your previous action sequence is {prev_actions}.
    Your current observation is {current_observation}.
    Your current height is {-int(current_position.z_val)} meters.
    Based on your current sub-goal and your current observation, whether you achieve the sub-goal?
    Your should answer 'YES' or 'NO', and provide the reason why you give such an answer.
    
    Example output: 
    YES: the drone already reached a high altitude.
    NO: the landmark in the sub-goal is not observed.
    """
    return prompt_str

def cot_prompt_builder_p3(
        navigation_instruction,
        prev_actions,
        current_subgoal,
        subgoal_status,
        current_observation,
):
    if subgoal_status is True:
        prompt_str = f"""
        You are an AI agent helping to control the drone to finish the navigation task.
        Your navigation task is {navigation_instruction}.
        So far, you've reach the sub-goal: {current_subgoal}.
        Based on your reached sub-goal and your navigation task, what's your next sub-goal to reach?
        
        
        """
    else:
        action_list = [act for act in DefaultAirsimActionCodes]
        prompt_str = f"""
        You are an AI agent helping to control the drone to finish the navigation task.
        Your navigation task is {navigation_instruction}.
        You need to reach the sub-goal: {current_subgoal}.
        Your current observation is: {current_observation};
        your previous action sequence is: {prev_actions}
        Based on your sub-goal, current observation and previous actions, 
        please provide the next action you should take. You can ONLY use ONE of the following actions: {action_list}. 
        Also, you need to provide the reason why you take the action.
        
        Example output: 
        MOVE_FORWARD: the building in front appeared for a lot of times with the target, I think it's helpful for finding the target.
        MOVE_LEFT: the target building is on the left
        MOVE_FORWARD: this place is arrived in earlier experience, so you should follow the earlier path.
        STOP: you have arrived at the target.
        GO_UP: follow the navigation instruction, take off
        GO_DOWN: the goal is reached, land to the floor.
        """

    return prompt_str



def open_ended_action_manager_prompt_builder_v2(
        navigation_instruction,
        current_text,
        prev_action=None,
        experience=None,
        relate_knowledge=None,
        fire=False,
        exclude_actions=None,
):
    action_list = [act for act in DefaultAirsimActionCodes]
    prompt_str = f"""
    You are an AI agent helping to control the drone to finish the navigation task.
    Your navigation task is {navigation_instruction}.
    Your current observation is {current_text}.
    Your previous action sequence is {prev_action}
    Based on your navigation task, current observation and previous action, please provide the next action you should take. You can ONLY use ONE of the following actions: {action_list}. Also, you need to provide the reason why you take the action.
    
    Example output: 
    MOVE_FORWARD: the building in front appeared for a lot of times with the target, I think it's helpful for finding the target.
    MOVE_LEFT: the target building is on the left
    MOVE_FORWARD: this place is arrived in earlier experience, so you should follow the earlier path.
    STOP: you have arrived at the target.
    GO_UP: follow the navigation instruction, take off
    GO_DOWN: the goal is reached, land to the floor.

    """

    return prompt_str


def subtask_action_manager_prompt_builder(subtask, finished_checkpoint, ongoing_checkpoint, pano_observation):
    action_list = [act for act in DefaultAirsimActionCodes]
    action_list = action_list[:-2]

    prompt_str = f"""
    You are an AI agent helping to control the drone to finish the navigation task.
    Your navigation task is {subtask}.
    The checkpoints you have finished are: {finished_checkpoint}.
    The ongoing checkpoint is: {ongoing_checkpoint}.
    Your current observation is {pano_observation}.
    
    Based on your navigation task progress and current observation, please provide the next action you should take.
    You can ONLY use ONE of the following actions: {action_list}. Also, you need to provide the reason why you take the action.
    
    Example output: 
    MOVE_FORWARD: the building in front appeared for a lot of times with the target, I think it's helpful for finding the target.
    MOVE_LEFT: the target building is on the left
    MOVE_FORWARD: this place is arrived in earlier experience, so you should follow the earlier path.
    STOP: you have arrived at the target.
    GO_UP: follow the navigation instruction, take off
    GO_DOWN: the goal is reached, land to the floor.
    """

    return prompt_str

def summarize_view_prompt_builder(full_view):
    assert len(full_view) == len(ObservationDirections)
    view_str = ""
    for i in range(len(full_view)):
        view_str += f"{ObservationDirections[i]}: {full_view[i]}\n"
    prompt_str = f"""
    You are an AI agent helping to control the drone to navigate the street view. 
    Your task is to summarize the view of the environment. You are told the view of the environment for 8 directions. Please summarize the following view into 1 paragraph, to reveal the overview of the current place.
    {view_str}
    """
    return prompt_str


def summarize_view_observation(full_view, collision_risk=[]):
    collision_risk = ["no" for _ in range(len(full_view))]

    assert len(full_view) == len(ObservationDirections)
    view_str = ""
    for i in range(len(full_view)):
        view_str += f"{ObservationDirections[i]}: {full_view[i]}, collision risk: {collision_risk[i]}\n"

    return view_str


def relative_spatial_prompt_builder(path):
    if len(path) == 0:
        return ""

    curr_loc = path[0]
    prev_loc = path[1]
    distance = path[2]
    rel_direction = path[3]
    prompt_str = f"{curr_loc} is located {distance} meters {rel_direction} of the {prev_loc} "

    return prompt_str


def landmark_memory_prompt_builder(instruction, landmarks):
    landmark_str = "\n".join(landmarks)
    landmark_path = []
    landmark_path_strs = []

    cnt = 1
    for i in range(len(landmarks)-1):
        # for j in range(i+1, len(landmarks)):
        #     landmark_path.append(f"{cnt}. <{landmarks[i]}> to <{landmarks[j]}>")
        #     cnt += 1
        landmark_path_strs.append(f"{cnt}. <{landmarks[i]}> to <{landmarks[i+1]}>")
        landmark_path.append([landmarks[i], landmarks[i+1]])
        cnt += 1

    landmark_path_prompt = "\n".join(landmark_path_strs)

    prompt = f"""
    Navigation instruction: {instruction}
    
    Landmarks in the instruction: 
    {landmark_str}
    
    Based on the instruction, describe the path between following path:
    {landmark_path_prompt}
    
    Your output format:
    1. <path>
    2. <path>
    ...
    """

    return prompt, landmark_path


def landmark_caption_prompt_builder(scene_objects):
    if len(scene_objects) == 0:
        prompt = f"""
List the objects that appears in the images. Each object use no more than 5 words to describe.

Example output:
object1.object2.object3
"""
    else:
        prompt = f"""
List the objects that appears in the image from the list below:
{scene_objects}.

Example output:
object1.object2.object3

Your output:
    """

    return prompt


def route_planning_prompt_builder(observed_objects, navigation_instruction, next_subgoal):
    prompt = f"""
You are a drone and your task is navigating to the described target location!

Navigation instruction: {navigation_instruction}

Your next navigation subgoal: {next_subgoal}

Objects or areas you observed: {observed_objects}

Based on the instruction, next navigation subgoal and observation, list 3 objects you will probably go next from your OBSERVED OBJECTS in descending order of probability. 

Example output:
object1.object2.object3

Your output:

    """

    return prompt


def prompt_updator_v2(original_prompt, ongoing_task=None, action_code=None, observations=None, action_seq_num=1):
    ori_prompt_splits = original_prompt.split("\n\n")

    intro_text = ori_prompt_splits[0]
    action_space_text = ori_prompt_splits[1]
    obs_direc_text = ori_prompt_splits[2]
    navi_instruction_text = ori_prompt_splits[3]
    action_obs_text = ori_prompt_splits[4]
    action_predict_prompt = ori_prompt_splits[5]


    # action_seq = "\n".join(ori_prompt_splits[14:-3])
    action_obs_seq = action_obs_text.split("\n")
    action_obs_seq = action_obs_seq[1:]

    if len(action_obs_seq) > 0:
        pattern = re.compile(r'^\d+')
        action_seq_num = 0
        for i in range(len(action_obs_seq) - 1, -1, -1):
            m = re.match(pattern, action_obs_seq[i])
            if m:
                action_seq_num = int(m.group()) + 1
                break
        if not action_seq_num:
            action_seq_num = 1
        action_obs_seq_text = "\n".join(action_obs_seq)
    else:
        action_obs_seq_text = ""


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
        if action_obs_seq_text != "":
            action_obs_seq_text = action_obs_seq_text+"\n"+f"{action_seq_num}. {action_str}"
        else:
            action_obs_seq_text = f"{action_seq_num}. {action_str}"

    observation_str = ""
    if observations:
        for landmark in observations:
            coarse_grained_loc, fine_grained_loc = observations[landmark]
            landmark_obs_str = f"There is {landmark} on the {fine_grained_loc} side of your {coarse_grained_loc} view.\n"
            observation_str += landmark_obs_str
    observation_str = observation_str.strip("\n")

    if observation_str != "":
        if action_obs_seq_text != "":
            action_obs_seq_text = action_obs_seq_text + "\n" + observation_str
        else:
            action_obs_seq_text = action_obs_seq_text + observation_str

    action_obs_text = "Action Sequence:\n"+action_obs_seq_text

    prompt = f"""{intro_text}

{action_space_text}

{obs_direc_text}

{navi_instruction_text}

{action_obs_text}

{action_predict_prompt}"""

    return prompt


def action_parser(action_str):
    action_str = action_str.lower()
    action_code = -1
    if "stop" in action_str:
        action_code = 0
    elif "forward" in action_str or "move forward" in action_str:
        action_code = 1
    elif "left" in action_str or "turn left" in action_str:
        action_code = 2
    elif "right" in action_str or "turn right" in action_str:
        action_code = 3
    elif "up" in action_str or "go up" in action_str:
        action_code = 4
    elif "down" in action_str or "go down" in action_str:
        action_code = 5

    return action_code


if __name__ == "__main__":
    navigation_prompt = """You are a drone and your task is navigating to the described target location!

Action Space: MOVE FORWARD, TURN LEFT, TURN RIGHT, GO UP, GO DOWN, STOP . The angle you turn left or right each time is 15 degrees.

Observation Direction: 
front: facing directly to the front
left: facing directly to the left
right: facing directly to the right
slightly left: 45 degrees to the front left
slightly right: 45 degrees to the front right

Navigation Instructions:
"Fly along the road until you see a red telephone booth on the center of your left. Turn left when you see the road on your left. Go up until you see a red billboard with a bull on it."

Action Sequence:

<Your next immediate action>"""
    obs = {"road": ["front", "top left"], "tree": ["left", "bottom down"]}
    new_prompt = prompt_updator_v2(navigation_prompt, observations=obs)
    print(new_prompt)
    print("-----------")
    new_prompt = prompt_updator_v2(new_prompt, action_code=1)
    print(new_prompt)
    print("-----------")
    new_prompt = prompt_updator_v2(new_prompt, observations=obs)
    print(new_prompt)
    print("-----------")
    new_prompt = prompt_updator_v2(new_prompt, action_code=2)
    print(new_prompt)

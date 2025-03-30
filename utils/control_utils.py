import sys
sys.path.append("..")
from airsim_plugin.airsim_settings import DefaultAirsimActionCodes

def action_str2enum(action_str):
    if "FORWARD" in action_str:
        return DefaultAirsimActionCodes["MOVE_FORWARD"]
    elif "TURN_LEFT" in action_str or "TURN LEFT" in action_str:
        return DefaultAirsimActionCodes["TURN_LEFT"]
    elif "TURN_RIGHT" in action_str or "TURN RIGHT" in action_str:
        return DefaultAirsimActionCodes["TURN_RIGHT"]
    elif "UP" in action_str:
        return DefaultAirsimActionCodes["GO_UP"]
    elif "DOWN" in action_str:
        return DefaultAirsimActionCodes["GO_DOWN"]
    elif "MOVE_LEFT" in action_str or "MOVE LEFT" in action_str:
        return DefaultAirsimActionCodes["MOVE_LEFT"]
    elif "MOVE_RIGHT" in action_str or "MOVE RIGHT" in action_str:
        return DefaultAirsimActionCodes["MOVE_RIGHT"]
    elif "STOP" in action_str:
        return DefaultAirsimActionCodes["STOP"]
    else:
        raise ValueError("Unknown action type: {}".format(action_str))



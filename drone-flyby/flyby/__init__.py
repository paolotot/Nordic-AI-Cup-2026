"""Our drone-flyby solution: detector -> tracker -> camera policy.

Settings come from environment variables so the server needs no code change:

    FLYBY_DETECTOR  yolo (default) | oracle   oracle = helsinki ground truth, testing only
    FLYBY_WEIGHTS   path to a .pt or *_openvino_model folder (default: models/detector*)
    FLYBY_CAMERA    look-and-zoom (default) | tour
"""

"""Entry point the server calls. The solution itself lives in the flyby package.

The pipeline (and the detector's warm-up) is built when api.py imports this
module, i.e. at server start, so the first scored frame is not the slow one.
See flyby/__init__.py for the environment variables that pick the detector
and the camera policy.
"""

from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from flyby.pipeline import Pipeline

_pipeline = Pipeline()


def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    return _pipeline.predict(request)

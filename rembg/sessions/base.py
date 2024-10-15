import os

import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL.Image import Image as PILImage
import onnxruntime as ort

class BaseSession:
    def __init__(self, model_name, sess_opts: ort.SessionOptions,
        providers=None,
        *args,
        **kwargs):
        self.model_name = model_name
        self.providers = []

        _providers = ort.get_available_providers()
        if providers:
            for provider in providers:
                if provider in _providers:
                    self.providers.append(provider)
        else:
            self.providers.extend(_providers)

        self.inner_session = ort.InferenceSession(os.path.join(self.u2net_home(), self.model_name + ".onnx"), providers=self.providers,sess_options=sess_opts,)

    @classmethod
    def u2net_home(cls):
        return os.path.expanduser(
            os.getenv("U2NET_HOME", os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".u2net"))
        )
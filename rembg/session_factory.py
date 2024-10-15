import onnxruntime as ort

from .sessions.u2net import U2NetSession

def new_session( model_name: str = "u2net", providers=None, *args, **kwargs):
    session_class = U2NetSession

    sess_opts = ort.SessionOptions()

    return session_class(model_name, sess_opts, providers, *args, **kwargs)
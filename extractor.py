from torchreid.reid.utils import FeatureExtractor

REID_MODEL = "osnet_x0_5"
DEVICE = "cpu"
WEIGHTS_PATH = "models/osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"

class Extractor:
    # model = None

    def __init__(self):
        self.model = FeatureExtractor(
            model_name=REID_MODEL,
            model_path=str(WEIGHTS_PATH),
            device=DEVICE,
            verbose=False,
        )

    def get_model(self):
        return self.model

    def get_model_name(self):
        return REID_MODEL
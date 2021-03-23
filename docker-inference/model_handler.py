import os
import io
import json
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import hydra
import logging
from omegaconf import OmegaConf
import os
import sys


DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

log = logging.getLogger(__name__)

# Import building function for model and dataset
from torch_points3d.datasets.dataset_factory import instantiate_dataset, get_dataset_class
from torch_points3d.models.model_factory import instantiate_model

# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

# Import from metrics
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

# Utils import
from torch_points3d.utils.colors import COLORS
                
class PyTorch3dPoint():
    def __init__(self):

        self.checkpoint_file_path = None
        self.model = None
        self.mapping = None
        self.device = "cpu"
        self.initialized = False
        self.model_name = None
        self.weight_name = None

    def initialize(self, context):
        """
           Load the model and mapping file to perform infernece.
        """

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        if not os.path.exists('/opt/ml/input'):
            os.makedirs('/opt/ml/input')
        
        print(model_dir)
        print(os.listdir(model_dir))
        
        # Load training configuration
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
            self.model_name = config.get('model_name', 'pointnet2_charlesssg')
            self.weight_name = config.get('weight_name', 'miou')
            self.forward_category = config.get('forward_category', 'Cap')

        print('config', config)
        print('model_name', self.model_name)
        print('forward_category', self.forward_category)

        # Read checkpoint file
        checkpoint_file_path = os.path.join(model_dir, "{}.pt".format(self.model_name))
        if not os.path.isfile(checkpoint_file_path):
            raise RuntimeError("Missing model.pth file.")

        # Prepare the model 
        checkpoint = ModelCheckpoint(model_dir, self.model_name, self.weight_name, strict=True)
        self.checkpoint = checkpoint
        
        print('checkpoint data_config', checkpoint.data_config)
        
        train_dataset_cls = get_dataset_class(self.checkpoint.data_config)
        setattr(self.checkpoint.data_config, "class", train_dataset_cls.FORWARD_CLASS)
        setattr(self.checkpoint.data_config, "forward_category", self.forward_category)
        
        self.initialized = True


    def forward_pass(self, model: BaseModel, dataset: BaseDataset, device, output_path):
        loaders = dataset.test_dataloaders
        predicted = {}
        for loader in loaders:
            loader.dataset.name
            with Ctq(loader) as tq_test_loader:
                for data in tq_test_loader:
                    with torch.no_grad():
                        model.set_input(data, device)
                        model.forward()
                    predicted = {**predicted, **dataset.predict_original_samples(data, model.conv_type, model.get_output())}
        return predicted

    def inference(self, data, context):
        input_dir = "/opt/ml/input/{}/".format(context.get_request_id())
        if not os.path.isdir(input_dir):
            os.mkdir(input_dir)
        with open("{}/inf_file.txt".format(input_dir), "w") as f:
            f.write(data[0]['body'].decode())
            
        data_config = self.checkpoint.data_config.copy()
        setattr(data_config, "dataroot", '/opt/ml/input/{}'.format(context.get_request_id()))
        
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        log.info("DEVICE : {}".format(device))

        # Enable CUDNN BACKEND
        torch.backends.cudnn.enabled = False       

        # Datset specific configs
        dataset = instantiate_dataset(self.checkpoint.data_config)
        model = self.checkpoint.create_model(dataset, weight_name=self.weight_name)
        log.info(model)
        log.info("Model size = %i", sum(param.numel() for param in model.parameters() if param.requires_grad))

        # Set dataloaders (model, batch size, shuffle)
        dataset.create_dataloaders(
            model, 1, True, 4, False,
        )
        log.info(dataset)

        model.eval()
        model = model.to(device)

        # Run training / evaluation
        if not os.path.exists('/opt/ml/output'):
            os.makedirs('/opt/ml/output')

        prediction = self.forward_pass(model, dataset, device, '/opt/ml/output')
        os.remove("{}/inf_file.txt".format(input_dir))
        os.rmdir(input_dir)
        return prediction
    
    def postprocess(self, inference_output):
        results = {}
        predictions = next(iter(inference_output.values())).tolist()
        results['response'] = predictions
        return  json.dumps(results)


_service = PyTorch3dPoint()
def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None
    
    #print('input data', data)
    
    data = _service.inference(data, context)
    results = _service.postprocess(data)

    return [results]

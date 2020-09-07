import torch.nn as nn
from .config import config_for_learning                       #local
import torch

from flask import current_app

try:
    from flask import _app_ctx_stack as stack
except ImportError:
    from flask import _request_ctx_stack as stack
    
    
###
'''프로젝트에 사용된 모델'''
class CNN(nn.Module) :
    def __init__(self, sentence_len, embedding_dim=100, num_filters=100, kernel_size=[3,4,5],stride=1, output_size=3):
        super(CNN, self).__init__()
        self.embedding_dim = embedding_dim

        conv_output_size1 = int((sentence_len - kernel_size[0]) / stride) + 1  # first conv layer output size
        conv_output_size2 = int((sentence_len - kernel_size[1]) / stride) + 1  # first conv layer output size
        conv_output_size3 = int((sentence_len - kernel_size[2]) / stride) + 1  # first conv layer output size

        # three convolution & pooling layers
        self.conv1 = nn.Conv2d(1, 100, kernel_size=(kernel_size[0], self.embedding_dim), stride=stride,padding=0)
        self.pool1 = nn.MaxPool2d((conv_output_size1, 1))  # Max-over-time pooling
        self.conv2 = nn.Conv2d(1, 100, kernel_size=(kernel_size[1], self.embedding_dim), stride=stride,padding=0)
        self.pool2 = nn.MaxPool2d((conv_output_size2, 1))  # Max-over-time pooling
        self.conv3 = nn.Conv2d(1, 100, kernel_size=(kernel_size[2], self.embedding_dim), stride=stride,padding=0)
        self.pool3 = nn.MaxPool2d((conv_output_size3, 1))  # Max-over-time pooling

        self.relu = nn.ReLU()
        self.dense = nn.Linear(num_filters * len(kernel_size),output_size)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1, self.embedding_dim)  # resize to fit into convolutional layer
        x1_pre = self.relu(self.conv1(x))
        x2_pre = self.relu(self.conv2(x))
        x3_pre = self.relu(self.conv3(x))

        x1 = self.pool1(x1_pre)
        x2 = self.pool2(x2_pre)
        x3 = self.pool3(x3_pre)
        x_pre = torch.cat((x1_pre,x2_pre,x3_pre),dim = 2)
        x = torch.cat((x1, x2, x3), dim=1)  # concatenate three convolutional outputs
        x = x.view(x.size(0), -1)  # resize to fit into final dense layer
        x = self.dense(x)
        return x, x_pre


class FlaskDeepModel(object):
    
    _model = {}

    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        app.config.setdefault('MODEL_FILE', 'wordvectors.bin')
            
        # model
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        model = CNN(config_for_learning['embedding_dim'], config_for_learning['num_filters'], kernel_size=config_for_learning['kernel_sizes'], stride=1)


        model_path = app.config['model_path']

        if use_cuda:
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))

        model.to(device)
        model.eval()

        self._model[app.config['MODEL_FILE']] = model
        
        
            

    @property
    def CNN_Model(self):
        ctx = stack.top
        if ctx is not None:
            if not hasattr(ctx, '_model'):
                ctx._model = self._model[current_app.config['MODEL_FILE']]
            return ctx._model
        
dmodel = FlaskDeepModel()
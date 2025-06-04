import torch
import torch.nn as nn
from torchvision.models import resnet18
from transformers import BertModel

class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        # 图像模型
        self.image_model = resnet18(pretrained=True)
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, 256)
        
        # 文本模型
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.text_fc = nn.Linear(self.text_model.config.hidden_size, 256)
        
        # 融合层
        self.fusion_layer = nn.Linear(256 * 2, 128)
        self.output_layer = nn.Linear(128, 1)
    
    def forward(self, images, input_ids, attention_mask):
        # 图像特征提取
        image_features = self.image_model(images)
        
        # 文本特征提取
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_fc(text_output.pooler_output)
        
        # 特征融合
        combined_features = torch.cat((image_features, text_features), dim=1)
        fused_features = torch.relu(self.fusion_layer(combined_features))
        
        # 输出
        output = self.output_layer(fused_features)
        return output
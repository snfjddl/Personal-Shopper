import torch
import torch.nn as nn
import torchvision.models as models

from PIL import Image
from torchvision import transforms
import json

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

labels = json.load(open(r'C:\Users\tjrdl\Downloads\final_web\classification_labels.txt'))
label_dict = { }
filenames = list(labels)

for label_type in ['category']:
    label_dict[label_type] = { }
    label_list = list(set([ labels[f][label_type] for f in labels ]))
    label_list.sort()
    for idx, label in enumerate(label_list):
        label_dict[label_type][idx] = label
        label_dict[label_type][label] = idx

labels2 = json.load(open(r'C:\Users\tjrdl\Downloads\final_web\color_labels.txt'))
label_dict2 = { }
filenames2 = list(labels2)

for label_type in ['color']:
    label_dict2[label_type] = { }
    label_list = list(set([ labels2[f][label_type] for f in labels2 ]))
    label_list.sort()
    for idx, label in enumerate(label_list):
        label_dict2[label_type][idx] = label
        label_dict2[label_type][label] = idx

num_label = len(label_dict['category'])//2
num_label2 = len(label_dict2['color'])//2

trans = transforms.ToTensor()

def predict_img(img_filepath):
    my_model = models.resnet50(pretrained=True)
    my_model.fc = nn.Linear(my_model.fc.in_features, num_label)
    my_model.load_state_dict(
        torch.load(r'C:\Users\tjrdl\Downloads\final_web\final_classification_model.pth', map_location='cpu'))
    my_model.to(device)
    my_model.eval()

    x = trans(Image.open(img_filepath))[None,:]
    x = x.to(device)
    y = my_model(x)

    category = label_dict['category'][int(y.argmax(1))]

    my_model2 = models.resnet50(pretrained=True)
    my_model2.fc = nn.Linear(my_model2.fc.in_features, num_label2)
    my_model2.load_state_dict(
        torch.load(r'C:\Users\tjrdl\Downloads\final_web\color_model.pth', map_location='cpu'))
    my_model2.to(device)
    my_model2.eval()

    color = label_dict2['color'][int(y.argmax(1))]

    return category, color

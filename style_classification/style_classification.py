import torch
import torchvision.models as models
from torchvision import transforms
import json
from PIL import Image


def style_return(img):

    img_locate = img
    

    labels = json.load(open('./labels.txt'))
    label_dict = { }
    filenames = list(labels)

    for label_type in ['attribute']:
        label_dict[label_type] = { }
        label_list = list(set([ labels[f][label_type] for f in labels]))
        label_list.sort()
        for idx, label in enumerate(label_list):
            label_dict[label_type][idx] = label
            label_dict[label_type][label] = idx


    print(label_dict)

    model = models.resnet50(num_classes=4)  

    device = torch.device('cpu')

    path = 'classification_model.pt(모델위치)'

    model.load_state_dict(torch.load(path, map_location=device))
    
    trans = transforms.ToTensor()
    x = trans(Image.open(img_locate))[None,:]
    x = x.to(device)
    y = model(x)

    return label_dict[label_type][int(y.argmax(1))-1]

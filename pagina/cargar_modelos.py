import os
import torch
from torchvision import models
import torch.nn as nn

def load_pretrained_model(model_name: str, num_classes: int):
    model_name = model_name.lower()

    if model_name == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b4":
        model = models.efficientnet_b4(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "googlenet":
        model = models.googlenet(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "inception_v3":
        model = models.inception_v3(pretrained=False, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Modelo '{model_name}' no soportado")

    return model

def cargar_modelos_desde_directorio(directorio: str, num_clases: int):
    modelos_cargados = {}

    for archivo in os.listdir(directorio):
        if archivo.endswith(".pth"):
            nombre_modelo = archivo.replace(".pth", "")
            ruta = os.path.join(directorio, archivo)

            try:
                model = load_pretrained_model(nombre_modelo, num_clases)
                state_dict = torch.load(ruta, map_location='cpu')
                model.load_state_dict(state_dict)
                model.eval()
                modelos_cargados[nombre_modelo] = model
                print(f"✅ Modelo cargado: {nombre_modelo}")
            except Exception as e:
                print(f"❌ Error al cargar '{archivo}': {e}")

    return modelos_cargados

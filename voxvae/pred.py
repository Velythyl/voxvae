import json
import os

import torch

from voxvae.dataloading.dataloader import load_for_inference
from voxvae.load import load_vma
from voxvae.utils.wandb_hydra import load_wandbconfig_as_hydraconfig


VMA = None
CFG = None
CACHE = {}
DEVICE = None
STORAGE_DEVICE = None
def setup(vma_path, vma_checkpoint=-1, inference_device="cuda:0", storage_device="cpu"):
    global VMA
    global CFG
    global CACHE
    global DEVICE
    global STORAGE_DEVICE
    assert VMA is None
    assert CFG is None
    assert len(CACHE) == 0
    assert DEVICE is None
    assert STORAGE_DEVICE is None

    DEVICE = torch.device(inference_device)
    STORAGE_DEVICE = torch.device(storage_device)

    CFG = load_wandbconfig_as_hydraconfig(vma_path)

    VMA = load_vma(vma_path, vma_checkpoint, CFG).to(DEVICE)

    CACHE = {}


def xmlpath_2_jsonpath(xmlpath_or_jsonpath):
    if xmlpath_or_jsonpath.endswith('.xml'):
        return xmlpath_or_jsonpath.replace('.xml', '-parsed.json')
    assert xmlpath_or_jsonpath.endswith('-parsed.json')


def pred(robot_path, robot_component):
    robot_path = xmlpath_2_jsonpath(robot_path)
    if robot_path in CACHE:
        if robot_component in CACHE[robot_path]:
            return CACHE[robot_path][robot_component]

    jsonpaths, robotcomponents, batch = load_for_inference(robot_path, CFG)

    latents = VMA.get_latent(batch.to(DEVICE)).squeeze().to() # (B, L)

    for jsonpath, robot_component, latent in zip(jsonpaths, robotcomponents, latents):
        if jsonpath not in CACHE:
            CACHE[jsonpath] = {}
        assert robot_component not in CACHE[jsonpath]

        CACHE[jsonpath][robot_component] = latent

from tqdm import tqdm

def write_latents(root, vma_path, vma_checkpoint=-1, inference_device="cuda:0"):
    DEVICE = torch.device(inference_device)
    CFG = load_wandbconfig_as_hydraconfig(vma_path)
    VMA = load_vma(vma_path, vma_checkpoint, CFG).to(DEVICE)
    CACHE = {}


    json_paths = []
    for root, _, files in os.walk(root):
        for file in files:
            json_path = os.path.join(root, file)
            if json_path.endswith(".json") and "/metadata/" not in json_path:
                json_paths.append(json_path)

    for jsonpath in tqdm(json_paths):
        jsonpaths, robotcomponents, batch = load_for_inference(jsonpath, CFG)

        latents = VMA.get_latent(batch.to(DEVICE)).squeeze().tolist()  # (B, L)

        to_save = {
            robot_component: latent for robot_component, latent in zip(robotcomponents, latents)
        }

        with open(jsonpath.replace("-parsed.json), "r") as f:
        json.dump()

        for jsonpath, robot_component, latent in zip(jsonpaths, robotcomponents, latents):
            if jsonpath not in CACHE:
                CACHE[jsonpath] = {}
            assert robot_component not in CACHE[jsonpath]

            CACHE[jsonpath][robot_component] = latent



if __name__ == "__main__":
    setup("./wandb/run-20250506_132651-caqvr0m0")
    pred("/home/charlie/Desktop/MJCFConvert/mjcf2o3d/unimals_100/test/xml/floor-5506-0-7-01-15-34-13.xml", "limb/5")
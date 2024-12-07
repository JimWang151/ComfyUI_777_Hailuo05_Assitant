# Made by JimWang for ComfyUI
# 02/04/2023

import torch
import random
import folder_paths
import uuid
import json
import urllib.request
import urllib.parse
import os
import numpy as np

import xml.etree.ElementTree as ET
from lxml import etree

import comfy.utils
from comfy.cli_args import args

from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo
import collections
from torchvision.transforms import ToPILImage,ToTensor
import torchvision.transforms as T

from PIL import Image, ImageDraw

import collections

class Hailuo05:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frame_nums": ("INT", {"default":3}),
            },
        }

    RETURN_TYPES = ("JOB",)
    RETURN_NAMES = ("job",)
    FUNCTION = "Canvas_Init"
    OUTPUT_NODE = True
    CATEGORY = "Hailuo05"
    DESCRIPTION = "Basic config for your frame."


    def Canvas_Init(self, frame_nums):
        # 初始化画布高度和宽度
        if frame_nums < 1 or frame_nums > 4:
            num_images = 4
        if frame_nums == 1:
            image_sizes = [{'height': 768, 'width': 1024}]  # 规格1
        elif frame_nums == 2:
            image_sizes = [{'height': 768, 'width': 1024}, {'height': 640, 'width': 1024}]  # 规格1 和 规格2
        elif frame_nums == 3:
            image_sizes = [{'height': 768, 'width': 1024}, {'height': 512, 'width': 512}, {'height': 512, 'width': 512}]  # 1张规格1 和 2张规格3
        elif frame_nums == 4:
            image_sizes = [{'height': 768, 'width': 1024}, {'height': 640, 'width': 1024}, {'height': 512, 'width': 512}, {'height': 512, 'width': 512}]  # 1张规格1、1张规格2、2张规格3

        return (image_sizes,)


class ImgCombine:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_color": (("black", "blue","grey","white"), {"default": "black"}),
                 "fill_color": (("white", "black","blue","grey"), {"default": "white"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    INPUT_IS_LIST = (True,)
    FUNCTION = "stitch_images"
    CATEGORY = "Hailuo04"

    def preprocess_image(self,image):


        # 检查是否有批次维度 (e.g., [1, H, W, 3])
        if len(image.shape) == 4 and image.shape[0] == 1:
            image = image.squeeze(0)  # 移除批次维度

        # 检查是否是 [H, W, 3] 格式 (通道在最后)
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)  # 转换为 [3, H, W]

        # 如果图片是 float32 类型，归一化到 [0, 255] 并转换为 uint8
        if image.dtype == torch.float32:
            image = (image * 255).clamp(0, 255).to(torch.uint8)

        return image

    def add_outer_frame(self, image, border_width=20, frame_color=(255, 255, 255), perforation_diameter=15,
                        perforation_spacing=60,fill_color=(10, 186, 181)):
        """
        给图片添加边框样式

        """
        if isinstance(image, torch.Tensor):
            if image.max() <= 1.0:
                image = ToPILImage()(image)
            else:
                image = ToPILImage()(image / 255)

        img_width, img_height = image.size
        perforation_radius = perforation_diameter // 2
        canvas_width = img_width + 2 * border_width
        canvas_height = img_height + 2 * border_width
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        canvas.paste(image, (border_width, border_width))
        draw = ImageDraw.Draw(canvas)

        # 绘制黑色边框
        draw.rectangle(
            [(0, 0), (canvas_width - 1, canvas_height - 1)],
            outline=frame_color,
            width=border_width
        )

        # 绘制 perforation 样式
        for y in range(border_width + perforation_radius, canvas_height - border_width - perforation_radius,
                       perforation_diameter + perforation_spacing):
            draw.ellipse(
                [
                    (border_width // 2 - perforation_radius, y - perforation_radius),
                    (border_width // 2 + perforation_radius, y + perforation_radius)
                ],
                fill=fill_color
            )
            draw.ellipse(
                [
                    (canvas_width - border_width // 2 - perforation_radius, y - perforation_radius),
                    (canvas_width - border_width // 2 + perforation_radius, y + perforation_radius)
                ],
                fill=fill_color
            )
        for x in range(border_width + perforation_radius, canvas_width - border_width,
                       perforation_diameter + perforation_spacing):
            draw.ellipse(
                [
                    (x - perforation_radius, border_width // 2 - perforation_radius),
                    (x + perforation_radius, border_width // 2 + perforation_radius)
                ],
                fill=fill_color
            )
            draw.ellipse(
                [
                    (x - perforation_radius, canvas_height - border_width // 2 - perforation_radius),
                    (x + perforation_radius, canvas_height - border_width // 2 + perforation_radius)
                ],
                fill=fill_color
            )

        result = ToTensor()(canvas) * 255
        # 确保范围正确
        result = result.clamp(0, 255)

        return result

    def add_inner_frame(self, image, wline, direction, frame_width=6, frame_color=(255, 255, 255)):


        # 确保形状为 (3, H, W)
        if len(image.shape) == 4 and image.shape[-1] == 3:
            # 调整形状从 (1, H, W, C) 到 (C, H, W)
            image = image.permute(0, 3, 1, 2)[0]
        elif len(image.shape) != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected image shape (3, H, W), but got {image.shape}")

        frame_color = torch.tensor(frame_color, dtype=torch.uint8).view(3, 1, 1)
        _, height, width = image.shape
        framed_image = image.clone()

        if direction == "horizontal":
            framed_image[:, :, width//2 - frame_width // 2: width //2+ frame_width // 2] = frame_color
        elif direction == "vertical":
            framed_image[:, height - wline  - frame_width // 2: height - wline  + frame_width // 2, :] = frame_color
        else:
            raise ValueError("Invalid direction specified for inner frame.")


        return framed_image
    def color_mapping(self,color_str):

        color_map = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "blue": (10, 186, 181),
            "grey": (50, 50, 50),
            # 你可以在这里添加其他颜色映射
        }
        if isinstance(color_str, list):
            color_str = color_str[0]

        if isinstance(color_str, str):
            if color_str in color_map:
                color_str = color_map[color_str]
            else:
                raise ValueError(f"Unsupported color string: {color_str}")
        return color_str

    def stitch_images(self, images,frame_color,fill_color):

        frame_color = self.color_mapping(frame_color)
        fill_color = self.color_mapping(fill_color)

        for idx, img in enumerate(images):
            # 移除批次维度（如果存在）
            if img.shape[0] == 1:
                img = img.squeeze(0)  # 从 [1, H, W, C] -> [H, W, C]

            # 如果是 float32 数据，将其从 [0, 1] 映射到 [0, 255]
            if isinstance(img, torch.Tensor) and img.dtype == torch.float32:
                img = img.clamp(0, 1) * 255  # 从 [0, 1] 映射到 [0, 255]
                img = img.to(torch.uint8)  # 转换为 uint8 类型

        processed_images = []
        for img in images:
            img = self.preprocess_image(img)
            if len(img.shape) == 4 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)[0]  # 调整到 (C, H, W)
            elif len(img.shape) != 3 or img.shape[0] != 3:
                raise ValueError(f"Invalid image shape: {img.shape}")
            processed_images.append(img)

        # 分类图像
        vertical_images = [img for img in processed_images if img.shape[2] == 1024]
        horizontal_images = [img for img in processed_images if img.shape[2] == 512]

        # 检查图片数量和规格
        if len(images) == 1:
            # 只有1张图片，直接画边框
            final_image = processed_images[0]
        elif len(images) == 2:
            # 有2张图片，纵向拼接后再画边框
            final_image = torch.cat((processed_images[0], processed_images[1]), dim=1)
            _, wline,gg= processed_images[1].shape
            final_image = self.add_inner_frame(final_image, wline, direction="vertical",frame_color=frame_color)
        else:
            # 超过2张图片
            if len(horizontal_images) == 1:
                # 规格3的图片只有1张，提示图片结构错误
                raise ValueError("Invalid image structure: Only 1 image with size 512x512 found.")
            elif len(horizontal_images) == 2:
                # 规格3的图片有2张，横向拼接
                intermediate_image = torch.cat((horizontal_images[0], horizontal_images[1]), dim=2)
                intermediate_image = self.add_inner_frame(intermediate_image, 0, direction="horizontal",frame_color=frame_color)
            else:
                raise ValueError("Invalid image structure: More than 2 images with size 512x512 found.")

            # 依次将规格1、规格2（如果是3张图片时，规格2是没有的），和过度规格的图片进行纵向拼接
            if len(vertical_images) == 1:
                final_image = torch.cat((vertical_images[0], intermediate_image), dim=1)
                _, wline, gg = intermediate_image.shape
                final_image = self.add_inner_frame(final_image, wline, direction="vertical",frame_color=frame_color)
            elif len(vertical_images) == 2:
                final_image = torch.cat((vertical_images[0], vertical_images[1]), dim=1)
                _, wline, gg = vertical_images[1].shape
                final_image = self.add_inner_frame(final_image, wline, direction="vertical",frame_color=frame_color)
                print(f"First round inner frame")
                final_image = torch.cat((final_image, intermediate_image), dim=1)
                _, wline, gg = intermediate_image.shape
                final_image = self.add_inner_frame(final_image, wline, direction="vertical",frame_color=frame_color)
            else:
                raise ValueError("Invalid image structure: More than 2 images with size 1024x768 or 1024x640 found.")

        # 添加外边框


        result = self.add_outer_frame(final_image, 40,frame_color=frame_color,fill_color=fill_color)
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(result)
        result = result.permute(1, 2, 0)  # 从 [3, H, W] 变为 [H, W, 3]
        result = result.unsqueeze(0)  # 添加 batch 维度，变为 [1, H, W, 3]

        result_normalized = result / 255.0  # 正规化到 [0, 1]
        result_normalized = torch.clamp(result_normalized, 0.0, 1.0)  # 确保范围合法
        result_normalized = result_normalized.to(torch.float32)  # 转换为 float32

        # 返回归一化的 Tensor
        return (result_normalized,)


class GetBaPrompt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key_word": ("STRING", {"multiline": True, "tooltip": "The text to be encoded."}),
                "features": ("STRING", {"multiline": True, "tooltip": "The text to be encoded."}),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
                "step": ("INT", {"default": 1}),
                "tempid": ("INT", {"default": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "get_ba_prompt"
    OUTPUT_NODE = True
    CATEGORY = "Hailuo04"
    DESCRIPTION = "Generate prompt."

    def get_ba_prompt(self, step, key_word, width, height,features,tempid):
        # 初始化画布高度和宽度
        desc = "full body"
        version = "man"
        prompt = ""
        if width == 512 and height == 512:
            desc = "half body"
            if step % 2 == 0:
                desc = "close up"
        resolution = str(width) + "×" + str(height)

        templateid="template"+str(tempid)
        prompt = self.find_ba_prompt(resolution, key_word,desc,templateid)
        prompt=self.add_feature(prompt,features)
        return (prompt,)

    def find_ba_prompt(self, resolution, key_word, shot, templateid):
        # 解析XML文件

        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_file_path = os.path.join(current_dir, 'data', 'BA_Template.xml')
        # 构建 XML 文件的相对路径
        # 解析XML文件
        try:
            tree = ET.parse(xml_file_path)
        except ET.ParseError as e:
            print(f"XML parsing error: {e}")
            return None

        root = tree.getroot()

        # 查找匹配的template节点
        template = root.find(f"./template[@id='{templateid}']")
        if template is None:
            print(f"Template with ID '{templateid}' not found.")
            return None

        # 查找匹配的details节点
        for details in template.findall('details'):
            if (details.find('resolution').text == resolution and
                    details.find('key_word').text == key_word and
                    details.find('shot').text == shot):
                prompt = details.find('prompt').text
                return prompt

        # 如果没有找到匹配的details节点，返回None
        print(
            f"No matching details found for resolution '{resolution}', key_word '{key_word}', shot '{shot}' in template '{templateid}'.")
        return "1 black woman"

    def add_feature(self, prompt, features):
        # 检查字符串长度是否足够
        comma_index = prompt.find(',')

        # 如果没有找到逗号，直接返回原始的prompt
        if comma_index == -1:
            return prompt

        # 在第一个逗号后面插入features
        result = prompt[:comma_index + 1] + features + prompt[comma_index + 1:]

        return result


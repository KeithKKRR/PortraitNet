import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms


def get_probability(prob):
    return np.random.uniform(0, 1) < prob


class DataAugmentation:
    image_avg = torch.Tensor([123.68, 116.78, 103.94])
    image_val = 0.017

    @staticmethod
    def process_image(img, mask, prob=0.5, rotation_range=[-45, 45], resize_scale_range=[0.5, 1.5],
                      translation_range=[-0.25, 0.25], gaussian_sigma=10, blur_kernel_sizes=[3, 5],
                      color_change_range=[0.4, 1.7], brightness_change_range=[0.4, 1.7],
                      contrast_change_range=[0.6, 1.5], sharpness_change_range=[0.8, 1.3]):
        W, H = img.size
        original_img = img

        # random horizontal flip
        if get_probability(prob):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # random translation
        if get_probability(prob):
            translation_amount = np.random.uniform(translation_range[0], translation_range[1], 2)
            x1, y1, x2, y2 = 0, 0, W, H
            x1 = int(max(x1 + W * translation_amount[0], 0))
            y1 = int(max(y1 + H * translation_amount[1], 0))
            x2 = int(min(x2 + W * translation_amount[0], W))
            y2 = int(min(y2 + H * translation_amount[1], H))
            temp_img = Image.new("RGB", (W, H))
            temp_img.paste(img.crop([x1, y1, x2, y2]), [x1, y1])
            img = temp_img
            temp_img = Image.new("1", (W, H))
            temp_img.paste(mask.crop([x1, y1, x2, y2]), [x1, y1])
            mask = temp_img

        # random rotation
        if get_probability(prob):
            rotated_angle = np.random.uniform(rotation_range[0], rotation_range[1])
            img = img.rotate(rotated_angle)
            mask = mask.rotate(rotated_angle)

        # random resize
        if get_probability(prob):
            resize_scale = np.random.uniform(resize_scale_range[0], resize_scale_range[1])
            W, H = int(W * resize_scale), int(H * resize_scale)
            img = img.resize((W, H))
            mask = Image.fromarray(np.array(mask.resize((W, H))))

        # random Gaussian noise
        if get_probability(prob):
            noise = np.random.normal(0, gaussian_sigma, (H, W, 3))
            img = Image.fromarray(np.uint8(np.clip(np.array(img) + noise, 0, 255)))

        # random blur (from official code)
        if get_probability(prob):
            select = np.random.uniform(0, 1)
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            if select < 0.3:
                kernel_size = random.choice(blur_kernel_sizes)
                image = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            elif select < 0.6:
                kernel_size = random.choice(blur_kernel_sizes)
                image = cv2.medianBlur(img, kernel_size)
            else:
                kernel_size = random.choice(blur_kernel_sizes)
                image = cv2.blur(img, (kernel_size, kernel_size))
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # random color & brightness & contrast & sharpness change
        if get_probability(prob):
            color_factor = np.random.uniform(color_change_range[0], color_change_range[1])
            img = ImageEnhance.Color(img).enhance(color_factor)
            brightness_factor = np.random.uniform(brightness_change_range[0], brightness_change_range[1])
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)
            contrast_factor = np.random.uniform(contrast_change_range[0], contrast_change_range[1])
            img = ImageEnhance.Contrast(img).enhance(contrast_factor)
            sharpness_factor = np.random.uniform(sharpness_change_range[0], sharpness_change_range[1])
            img = ImageEnhance.Sharpness(img).enhance(sharpness_factor)

        # resize to 224
        if W >= H:
            H = int(H * 224 / W)
            W = 224
            img, mask = transforms.Resize([H, W])(img), transforms.Resize([H, W])(mask)
            img = transforms.Pad(padding=[0, (224 - H) // 2, 0, (224 - H) // 2 + (224 - H) % 2],
                                 padding_mode="constant")(img)
            mask = transforms.Pad(padding=[0, (224 - H) // 2, 0, (224 - H) // 2 + (224 - H) % 2],
                                  padding_mode="constant")(mask)
        else:
            W = int(W * 224 / H)
            H = 224
            img, mask = transforms.Resize([H, W])(img), transforms.Resize([H, W])(mask)
            img = transforms.Pad(padding=[(224 - W) // 2, 0, (224 - W) // 2 + (224 - W) % 2, 0],
                                 padding_mode="constant")(img)
            mask = transforms.Pad(padding=[(224 - W) // 2, 0, (224 - W) // 2 + (224 - W) % 2, 0],
                                  padding_mode="constant")(mask)

        # get edge
        edge = np.zeros((224, 224), np.uint8)
        ret, binary = cv2.threshold(np.uint8(mask) * 255, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # RETR_EXTERNAL
        cv2.drawContours(edge, contours, -1, 1, 4)

        # return img, mask, edge  # show image test
        return torch.Tensor(np.array(original_img)), torch.Tensor(np.array(img)), torch.Tensor(np.array(mask)), \
            torch.Tensor(edge)

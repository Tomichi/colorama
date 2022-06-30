from json import loads
from os import path
import csv

import numpy as np


class Color:
    rgb_centroids = {}

    def __init__(self):
        dirname_path = path.dirname(__file__)
        filename = path.join(dirname_path, 'colors.json')
        with open(filename, mode='r', encoding='utf-8') as file:
            raw_colors = loads(file.read())
            self.rgb_centroids = self.convert_centroids(raw_colors)

    @staticmethod
    def convert_hex2rgb(hex_color: str) -> np.ndarray:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            print(hex_color)
            raise RuntimeError('Bad format of hex color')

        hex_color = hex_color.upper()
        return np.array([float(int(hex_color[i:i + 2], 16) / 255.0) for i in range(0, 6, 2)])

    @staticmethod
    def convert_centroids(rgb_centroids: dict) -> dict:
        result = {}
        for color_name, colors in rgb_centroids.items():
            result[color_name] = []
            for hex_code in colors:
                result[color_name].append(Color.convert_hex2rgb(hex_code))
        return result



    def get_data(self):
        return self.rgb_centroids

    def data_to_csv(self):
        data = []
        for color_name, colors in self.rgb_centroids.items():
            for color in colors:
                data.append([str(color[0]), str(color[1]), str(color[2]), color_name])
        with open(file='rgb_color.csv', mode='w', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)


    def closed_color(self, unknown_hex_color: str) -> str:
        unknown_hex_color = unknown_hex_color.upper().lstrip('#')
        unknown_vector = self.convert_hex2rgb(unknown_hex_color)
        smallest_color_distance = 5.0
        smallest_color_name = None
        for color_name, color_vectors in self.rgb_centroids.items():
            for color in color_vectors:
                distance = np.linalg.norm(unknown_vector - color)
                if distance < smallest_color_distance:
                    smallest_color_distance = distance
                    smallest_color_name = color_name

        return smallest_color_name
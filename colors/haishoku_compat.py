"""In-memory Haishoku-compatible color extraction helpers."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from PIL import Image

ColorCount = Tuple[int, Tuple[int, int, int]]
PaletteEntry = Tuple[float, Tuple[int, int, int]]


def _get_thumbnail(image: Image.Image) -> Image.Image:
    thumbnail = image.copy()
    thumbnail.thumbnail((256, 256))
    if thumbnail.mode != "RGB":
        thumbnail = thumbnail.convert("RGB")
    return thumbnail


def get_colors(image: Image.Image) -> List[ColorCount]:
    """Return color counts using the same thumbnail-first strategy as Haishoku."""
    thumbnail = _get_thumbnail(image)
    max_colors = thumbnail.width * thumbnail.height
    colors = thumbnail.getcolors(max_colors)
    return colors or []


def sort_by_rgb(colors_tuple: Sequence[ColorCount]) -> List[ColorCount]:
    return sorted(colors_tuple, key=lambda x: x[1])


def rgb_maximum(colors_tuple: Sequence[ColorCount]) -> dict:
    r_sorted_tuple = sorted(colors_tuple, key=lambda x: x[1][0])
    g_sorted_tuple = sorted(colors_tuple, key=lambda x: x[1][1])
    b_sorted_tuple = sorted(colors_tuple, key=lambda x: x[1][2])

    r_min = r_sorted_tuple[0][1][0]
    g_min = g_sorted_tuple[0][1][1]
    b_min = b_sorted_tuple[0][1][2]

    r_max = r_sorted_tuple[-1][1][0]
    g_max = g_sorted_tuple[-1][1][1]
    b_max = b_sorted_tuple[-1][1][2]

    return {
        "r_max": r_max,
        "r_min": r_min,
        "g_max": g_max,
        "g_min": g_min,
        "b_max": b_max,
        "b_min": b_min,
        "r_dvalue": (r_max - r_min) / 3,
        "g_dvalue": (g_max - g_min) / 3,
        "b_dvalue": (b_max - b_min) / 3,
    }


def group_by_accuracy(sorted_tuple: Sequence[ColorCount], accuracy: int = 3):
    del accuracy
    rgb_maximum_json = rgb_maximum(sorted_tuple)
    r_min = rgb_maximum_json["r_min"]
    g_min = rgb_maximum_json["g_min"]
    b_min = rgb_maximum_json["b_min"]
    r_dvalue = rgb_maximum_json["r_dvalue"]
    g_dvalue = rgb_maximum_json["g_dvalue"]
    b_dvalue = rgb_maximum_json["b_dvalue"]

    rgb = [
        [[[], [], []], [[], [], []], [[], [], []]],
        [[[], [], []], [[], [], []], [[], [], []]],
        [[[], [], []], [[], [], []], [[], [], []]],
    ]

    for color_tuple in sorted_tuple:
        r_tmp_i = color_tuple[1][0]
        g_tmp_i = color_tuple[1][1]
        b_tmp_i = color_tuple[1][2]
        r_idx = 0 if r_tmp_i < (r_min + r_dvalue) else 1 if r_tmp_i < (r_min + r_dvalue * 2) else 2
        g_idx = 0 if g_tmp_i < (g_min + g_dvalue) else 1 if g_tmp_i < (g_min + g_dvalue * 2) else 2
        b_idx = 0 if b_tmp_i < (b_min + b_dvalue) else 1 if b_tmp_i < (b_min + b_dvalue * 2) else 2
        rgb[r_idx][g_idx][b_idx].append(color_tuple)

    return rgb


def get_weighted_mean(grouped_image_color: Sequence[ColorCount]) -> ColorCount:
    sigma_count = 0
    sigma_r = 0
    sigma_g = 0
    sigma_b = 0

    for item in grouped_image_color:
        sigma_count += item[0]
        sigma_r += item[1][0] * item[0]
        sigma_g += item[1][1] * item[0]
        sigma_b += item[1][2] * item[0]

    r_weighted_mean = int(sigma_r / sigma_count)
    g_weighted_mean = int(sigma_g / sigma_count)
    b_weighted_mean = int(sigma_b / sigma_count)

    return (sigma_count, (r_weighted_mean, g_weighted_mean, b_weighted_mean))


def get_colors_mean(image: Image.Image) -> List[ColorCount]:
    image_colors = get_colors(image)
    if not image_colors:
        return []

    sorted_image_colors = sort_by_rgb(image_colors)
    grouped_image_colors = group_by_accuracy(sorted_image_colors)

    colors_mean = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                grouped_image_color = grouped_image_colors[i][j][k]
                if grouped_image_color:
                    colors_mean.append(get_weighted_mean(grouped_image_color))

    temp_sorted_colors_mean = sorted(colors_mean)
    if len(temp_sorted_colors_mean) > 8:
        colors_mean = temp_sorted_colors_mean[-8:]
    else:
        colors_mean = temp_sorted_colors_mean

    return sorted(colors_mean, reverse=True)


def get_dominant(image: Image.Image) -> Tuple[int, int, int]:
    colors_mean = get_colors_mean(image)
    if not colors_mean:
        raise ValueError("Could not extract dominant color from image")
    return colors_mean[0][1]


def get_palette(image: Image.Image) -> List[PaletteEntry]:
    colors_mean = get_colors_mean(image)
    if not colors_mean:
        return []

    count_sum = sum(c_m[0] for c_m in colors_mean)
    if count_sum <= 0:
        return []

    palette = []
    for count, color in colors_mean:
        percentage = float("%.2f" % (count / count_sum))
        palette.append((percentage, color))

    return palette

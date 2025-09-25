from __future__ import annotations

from ..reading_order import prediction_reading_order
from ..utils.misc import calc_overlap_ratio, is_contained, quad_to_xyxy
from ..schemas import ParagraphSchema, FigureSchema


def combine_flags(flag1, flag2):
    return [f1 or f2 for f1, f2 in zip(flag1, flag2)]


def judge_page_direction(paragraphs):
    h_sum_area = 0
    v_sum_area = 0

    for paragraph in paragraphs:
        x1, y1, x2, y2 = paragraph.box
        w = x2 - x1
        h = y2 - y1

        if paragraph.direction == "horizontal":
            h_sum_area += w * h
        else:
            v_sum_area += w * h

    if v_sum_area > h_sum_area:
        return "vertical"

    return "horizontal"


def extract_paragraph_within_figure(paragraphs, figures):
    new_figures = []
    check_list = [False] * len(paragraphs)
    for figure in figures:
        figure = {"box": figure.box, "order": 0}
        contained_paragraphs = []
        for i, paragraph in enumerate(paragraphs):
            if is_contained(figure["box"], paragraph.box, threshold=0.7):
                contained_paragraphs.append(paragraph)
                check_list[i] = True

        figure["direction"] = judge_page_direction(contained_paragraphs)
        reading_order = (
            "left2right" if figure["direction"] == "horizontal" else "right2left"
        )

        figure_paragraphs = prediction_reading_order(
            contained_paragraphs, reading_order
        )
        figure["paragraphs"] = sorted(figure_paragraphs, key=lambda x: x.order)
        figure = FigureSchema(**figure)
        new_figures.append(figure)

    return new_figures, check_list


def extract_words_within_element(pred_words, element):
    contained_words = []
    word_sum_width = 0
    word_sum_height = 0
    check_list = [False] * len(pred_words)

    for i, word in enumerate(pred_words):
        word_box = quad_to_xyxy(word.points)
        if is_contained(element.box, word_box, threshold=0.5):
            word_sum_width += word_box[2] - word_box[0]
            word_sum_height += word_box[3] - word_box[1]
            check_list[i] = True

            word_element = ParagraphSchema(
                box=word_box,
                contents=word.content,
                direction=word.direction,
                order=0,
                role=None,
            )
            contained_words.append(word_element)

    if len(contained_words) == 0:
        return None, None, check_list

    word_direction = [word.direction for word in contained_words]
    cnt_horizontal = word_direction.count("horizontal")
    cnt_vertical = word_direction.count("vertical")

    element_direction = "horizontal" if cnt_horizontal > cnt_vertical else "vertical"
    order = "left2right" if element_direction == "horizontal" else "right2left"
    prediction_reading_order(contained_words, order)
    contained_words = sorted(contained_words, key=lambda x: x.order)

    contained_words = "\n".join([content.contents for content in contained_words])

    return (contained_words, element_direction, check_list)

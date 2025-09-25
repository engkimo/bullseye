from __future__ import annotations

from typing import List, Tuple

from .geometry import is_vertical, is_noise
from .misc import calc_overlap_ratio, is_contained, quad_to_xyxy


def extract_words_within_table(words, table, check_list: List[bool]):
    horizontal_words = []
    vertical_words = []

    for i, (points, score) in enumerate(zip(words.points, words.scores)):
        word_box = quad_to_xyxy(points)
        if is_contained(table.box, word_box, threshold=0.5):
            if is_vertical(points):
                vertical_words.append({"points": points, "score": score})
            else:
                horizontal_words.append({"points": points, "score": score})

            check_list[i] = True

    return horizontal_words, vertical_words, check_list


def calc_overlap_words_on_lines(lines, words) -> List[List[float]]:
    overlap_ratios: List[List[float]] = [[0 for _ in lines] for _ in words]

    for i, word in enumerate(words):
        word_box = quad_to_xyxy(word["points"])
        for j, row in enumerate(lines):
            overlap_ratio, _ = calc_overlap_ratio(row.box, word_box)
            overlap_ratios[i][j] = overlap_ratio

    return overlap_ratios


def correct_vertical_word_boxes(overlap_ratios_vertical, table, table_words_vertical):
    allocated_cols = [cols.index(max(cols)) for cols in overlap_ratios_vertical]

    new_points = []
    new_scores = []
    for i, col_index in enumerate(allocated_cols):
        col_cells = []
        for cell in table.cells:
            if cell.col <= (col_index + 1) < (cell.col + cell.col_span):
                col_cells.append(cell)

        word_point = table_words_vertical[i]["points"]
        word_score = table_words_vertical[i]["score"]

        for cell in col_cells:
            word_box = quad_to_xyxy(word_point)
            _, intersection = calc_overlap_ratio(cell.box, word_box)

            if intersection is not None:
                _, y1, _, y2 = intersection

                new_point = [
                    [word_point[0][0], max(word_point[0][1], y1)],
                    [word_point[1][0], max(word_point[1][1], y1)],
                    [word_point[2][0], min(word_point[2][1], y2)],
                    [word_point[3][0], min(word_point[3][1], y2)],
                ]

                if not is_noise(new_point):
                    new_points.append(new_point)
                    new_scores.append(word_score)

    return new_points, new_scores


def correct_horizontal_word_boxes(
    overlap_ratios_horizontal, table, table_words_horizontal
):
    allocated_rows = [rows.index(max(rows)) for rows in overlap_ratios_horizontal]

    new_points = []
    new_scores = []
    for i, row_index in enumerate(allocated_rows):
        row_cells = []
        for cell in table.cells:
            if cell.row <= (row_index + 1) < (cell.row + cell.row_span):
                row_cells.append(cell)

        word_point = table_words_horizontal[i]["points"]
        word_score = table_words_horizontal[i]["score"]

        for cell in row_cells:
            word_box = quad_to_xyxy(word_point)
            _, intersection = calc_overlap_ratio(cell.box, word_box)

            if intersection is not None:
                x1, _, x2, _ = intersection

                new_point = [
                    [max(word_point[0][0], x1), word_point[0][1]],
                    [min(word_point[1][0], x2), word_point[1][1]],
                    [min(word_point[2][0], x2), word_point[2][1]],
                    [max(word_point[3][0], x1), word_point[3][1]],
                ]

                if not is_noise(new_point):
                    new_points.append(new_point)
                    new_scores.append(word_score)

    return new_points, new_scores


def split_text_across_cells(results_det, results_layout):
    """Re-slice detected word quads when they cross table cell boundaries.

    Returns modified detection results with updated points/scores.
    """
    check_list = [False] * len(results_det.points)
    new_points = []
    new_scores = []
    for table in results_layout.tables:
        table_words_horizontal, table_words_vertical, check_list = (
            extract_words_within_table(results_det, table, check_list)
        )

        overlap_ratios_horizontal = calc_overlap_words_on_lines(
            table.rows,
            table_words_horizontal,
        )

        overlap_ratios_vertical = calc_overlap_words_on_lines(
            table.cols,
            table_words_vertical,
        )

        new_points_horizontal, new_scores_horizontal = correct_horizontal_word_boxes(
            overlap_ratios_horizontal, table, table_words_horizontal
        )

        new_points_vertical, new_scores_vertical = correct_vertical_word_boxes(
            overlap_ratios_vertical, table, table_words_vertical
        )

        new_points.extend(new_points_horizontal)
        new_scores.extend(new_scores_horizontal)
        new_points.extend(new_points_vertical)
        new_scores.extend(new_scores_vertical)

    for i, flag in enumerate(check_list):
        if not flag:
            new_points.append(results_det.points[i])
            new_scores.append(results_det.scores[i])

    results_det.points = new_points
    results_det.scores = new_scores
    return results_det


from __future__ import annotations

from typing import Dict, List

from .misc import filter_by_flag, is_contained


def filter_contained_rectangles_within_category(category_elements: Dict[str, List[dict]]):
    """Remove rectangles contained by others within the same category.

    Keeps the larger box when two boxes mutually contain each other (due to noise/rounding).
    """

    for category, elements in category_elements.items():
        group_box = [element["box"] for element in elements]
        check_list = [True] * len(group_box)
        for i, box_i in enumerate(group_box):
            for j, box_j in enumerate(group_box):
                if i >= j:
                    continue

                ij = is_contained(box_i, box_j)
                ji = is_contained(box_j, box_i)

                box_i_area = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                box_j_area = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])

                # If both claim containment (near-equal boxes), keep the larger area.
                if ij and ji:
                    if box_i_area > box_j_area:
                        check_list[j] = False
                    else:
                        check_list[i] = False
                elif ij:
                    check_list[j] = False
                elif ji:
                    check_list[i] = False

        category_elements[category] = filter_by_flag(elements, check_list)

    return category_elements


def filter_contained_rectangles_across_categories(
    category_elements: Dict[str, List[dict]], source: str, target: str
):
    """Remove target boxes if they are contained by any of source boxes."""

    src_boxes = [element["box"] for element in category_elements.get(source, [])]
    tgt_boxes = [element["box"] for element in category_elements.get(target, [])]

    check_list = [True] * len(tgt_boxes)
    for src_box in src_boxes:
        for j, tgt_box in enumerate(tgt_boxes):
            if is_contained(src_box, tgt_box):
                check_list[j] = False

    category_elements[target] = filter_by_flag(category_elements.get(target, []), check_list)
    return category_elements


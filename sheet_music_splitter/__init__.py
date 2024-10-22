from pathlib import Path
import sys
import time
import cv2
import numpy as np
import pypdfium2 as pdfium
import asyncio
import click
import json

"""
This is fairly basic.

I load up an image, then run OpenCV's line segment detector on it.
This gives me a bunch of short lines, which I then bound into rectangles with a bit of padding.
I then merge rectangles that overlap, and filter out rectangles that are too small.

Once this is done, there should be a few rectangles that represent the staff systems.

The one fault with this algorithm that I know of is that it doesn't
handle staff systems that are too close together.
It will merge them into one big rectangle.
"""


def compute_rectangles(line_sets):
    rects = []
    for line_set in line_sets:
        min_x = None
        min_y = None

        max_x = None
        max_y = None
        for line in line_set:
            if min_x is None:
                min_x = min(line[0], line[2])
            else:
                min_x = min(min_x, line[0], line[2])

            if min_y is None:
                min_y = min(line[1], line[3])
            else:
                min_y = min(min_y, line[1], line[3])

            if max_x is None:
                max_x = max(line[0], line[2])
            else:
                max_x = max(max_x, line[0], line[2])

            if max_y is None:
                max_y = max(line[1], line[3])
            else:
                max_y = max(max_y, line[1], line[3])

        rects.append(((min_x, min_y), (max_x, max_y)))
    return rects


class Point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def _tuple(self):
        return (self.x, self.y)


class Rectangle:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    @property
    def width(self):
        return self.p2.x - self.p1.x

    @property
    def height(self):
        return self.p2.y - self.p1.y

    @property
    def area(self):
        return self.width * self.height

    def _tuple(self):
        return (self.p1._tuple(), self.p2._tuple())

    def _flat_tuple(self):
        return (*self.p1._tuple(), *self.p2._tuple())

    def _json(self):
        return {"top_left": self.p1._tuple(), "bottom_right": self.p2._tuple()}


# DISCLAIMER: The following code is taken from GeeksForGeeks
# https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/


# Given three collinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
    if (
        (q.x <= max(p.x, r.x))
        and (q.x >= min(p.x, r.x))
        and (q.y <= max(p.y, r.y))
        and (q.y >= min(p.y, r.y))
    ):
        return True
    return False


def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.

    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if val > 0:
        # Clockwise orientation
        return 1
    elif val < 0:
        # Counterclockwise orientation
        return 2
    else:
        # Collinear orientation
        return 0


# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def check_intersection(line1, line2):
    l0 = (Point(line1[0], line1[1]), Point(line1[2], line1[3]))
    l1 = (Point(line2[0], line2[1]), Point(line2[2], line2[3]))

    p1, q1 = l0
    p2, q2 = l1

    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases

    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0) and onSegment(p1, p2, q1):
        return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0) and onSegment(p1, q2, q1):
        return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0) and onSegment(p2, p1, q2):
        return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0) and onSegment(p2, q1, q2):
        return True

    # If none of the cases
    return False


def isRectangleOverlap(rec1, rec2):
    """
    :type rec1: List[int]
    :type rec2: List[int]
    :rtype: bool
    """

    def intersect(p_left, p_right, q_left, q_right):
        return max(p_left, q_left) < min(p_right, q_right)

    return intersect(rec1[0], rec1[2], rec2[0], rec2[2]) and intersect(
        rec1[1], rec1[3], rec2[1], rec2[3]
    )


def rects_intersect(a, b):
    return isRectangleOverlap(a._flat_tuple(), b._flat_tuple())


def merge_rects(a, b):
    ax1, ay1, ax2, ay2 = a._flat_tuple()
    bx1, by1, bx2, by2 = b._flat_tuple()
    min_x = min(ax1, ax2, bx1, bx2)
    min_y = min(ay1, ay2, by1, by2)
    max_x = max(ax1, ax2, bx1, bx2)
    max_y = max(ay1, ay2, by1, by2)

    return Rectangle(Point(min_x, min_y), Point(max_x, max_y))


def convert_from_line(a, padding=0):
    ax1, ay1, ax2, ay2 = a
    min_x = max(min(ax1, ax2) - padding, 0)
    min_y = max(min(ay1, ay2) - padding, 0)
    max_x = max(ax1, ax2) + padding
    max_y = max(ay1, ay2) + padding
    return Rectangle(Point(min_x, min_y), Point(max_x, max_y))


def reduce_rectangles(rects):
    reduced_rects = []
    rects = sorted(rects, key=lambda r: r._tuple())

    while len(rects) > 0:
        current_rect = rects.pop(0)

        idx = 0
        while idx < len(rects):
            next_rect = rects[idx]
            if rects_intersect(current_rect, next_rect):
                current_rect = merge_rects(current_rect, next_rect)
                rects.pop(idx)
            else:
                idx += 1

        reduced_rects.append(current_rect)

    return reduced_rects


def get_page_rectangles(image_name, padding=3):
    img = cv2.imread(image_name)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 60, 150, apertureSize=3)
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=80, maxLineGap=9)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    lines = lsd.detect(gray)[0]

    lines = [tuple(int(a) for a in line[0]) for line in lines]

    _t0 = time.time()
    rectangles = [convert_from_line(line, padding=padding) for line in lines]

    tries = 10
    last_count = len(rectangles)
    print("start count", last_count)
    while tries > 0:
        tries -= 1

        rectangles = reduce_rectangles(rectangles)
        print("count rects", len(rectangles), file=sys.stderr)

        if last_count == len(rectangles):
            break

        last_count = len(rectangles)

    _t1 = time.time()

    print("time", _t1 - _t0, file=sys.stderr)

    avg_area = np.average([r.area for r in rectangles]) * 0.8
    rectangles = [r for r in rectangles if r.area >= avg_area]

    return rectangles


async def extract_pages(path: Path, pdf_file, key):
    pdf = pdfium.PdfDocument(pdf_file)

    path.mkdir(parents=True, exist_ok=True)

    files = []
    file_keys = []
    for i, page in enumerate(pdf):
        file_key = f"{key}-{i+1}.png"
        file_keys.append(file_key)
        file = path / file_key
        bitmap = page.render(scale=6, rotation=0)
        pil_image = bitmap.to_pil()
        pil_image.save(file)
        files.append(file)

    return files


@click.group()
def cli():
    pass


@cli.command()
@click.argument("pdf_file")
@click.argument("output_dir")
@click.option("--padding", default=3, help="Padding around the rectangles")
def rectangles(pdf_file, output_dir, padding=3):
    asyncio.run(_rectangles(pdf_file, output_dir, padding=padding))


async def _rectangles(pdf_file, output_dir, padding=3):
    path = Path(pdf_file)
    pages = await extract_pages(Path(output_dir), pdf_file, path.name)

    output = []

    for page in pages:
        rects = get_page_rectangles(page, padding=padding)
        output.append({"page": str(page), "rectangles": [r._json() for r in rects]})

    print(json.dumps(output, indent=2))


@cli.command()
@click.argument("pdf_file")
@click.argument("output_dir")
@click.option("--padding", default=3, help="Padding around the rectangles")
def images(pdf_file, output_dir, padding):
    asyncio.run(_images(pdf_file, output_dir, padding=padding))


async def _images(path, output_dir, padding=3):
    path = Path(path)
    pages = await extract_pages(Path(output_dir), path, path.name)

    for page in pages:
        rects = get_page_rectangles(page, padding=padding)
        img = cv2.imread(page)
        for rect in rects:
            cv2.rectangle(img, rect.p1._tuple(), rect.p2._tuple(), (0, 0, 255), 1)

        rects_file = page.parent / (page.stem + ".rects.png")
        cv2.imwrite(rects_file, img)

    print("Pages extracted, rectangles drawn in", output_dir)


if __name__ == "__main__":
    cli()

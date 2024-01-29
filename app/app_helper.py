import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (r'D:\Work\Uni\Artificial_intelligence\project\UML-to-Code\tesseract\tesseract.exe')

def get_class_text(image):
    def segment(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        markers = cv2.connectedComponents(thresh)[1]

        total_pixels = image.size

        segmented_regions = []

        for label in np.unique(markers):
            if label == -1:
                continue  # Skip the background

            mask = np.zeros_like(gray, dtype=np.uint8)
            mask[markers == label] = 255

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            non_black_pixels = cv2.countNonZero(mask)
            percentage_non_black = (non_black_pixels / total_pixels) * 100

            # Check if the region has more than 0.3% non-black pixels
            if percentage_non_black > 0.3:

                # save coordinates
                (x, y, width, height) = cv2.boundingRect(contours[0])
                if width > 5 and height > 5:

                    added = False
                    for other_region in segmented_regions:
                        if other_region[0][0] == x:
                            for subregion in other_region:
                                if subregion[1] + subregion[3] in range(y - 10, y + 10):
                                    other_region.append((x, y, width, height))
                                    added = True
                                    break
                    if not added:
                        segmented_regions.append([(x, y, width, height)])

        segmented_regions.pop(0)  # remove the whole diagram
        return segmented_regions

    text = ''
    segmented_regions = segment(image)
    for region in segmented_regions:
        for (x, y, w, h) in region:
            cropped_image = image[y:y + h, x:x + w]
            text += pytesseract.image_to_string(cropped_image) + '\n'
    return text
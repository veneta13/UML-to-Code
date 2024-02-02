import cv2
import numpy as np
import pytesseract

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/Mistral-7B-Code-16K-qlora-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

pytesseract.pytesseract.tesseract_cmd = (
    r'D:\Work\Uni\Artificial_intelligence\project\UML-to-Code\tesseract\tesseract.exe')


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

    def get_parameters(text):
        class_params = {}

        text = text.splitlines()
        text = [line for line in text if line != '']

        class_params['name'] = text[0]
        class_params['attributes'] = []
        class_params['methods'] = []

        for field in text:
            if field[0] == '+':
                class_params['attributes'].append(field[1:])
            elif '(' in field:
                class_params['methods'].append(field)

        return class_params

    text = ''
    segmented_regions = segment(image)
    for region in segmented_regions:
        for (x, y, w, h) in region:
            cropped_image = image[y:y + h, x:x + w]
            text += pytesseract.image_to_string(cropped_image) + '\n'

    return get_parameters(text)


def build_prompt(class_params):
    prompt = (f"Write a Python class with name:"
              f"{class_params['name']}, attributes: "
              f"{','.join(class_params['attributes'])} and methods: "
              f"{','.join(class_params['methods'])},"
              f"only filling the constructor and leaving the other methods with pass.")

    return prompt


def generate_code(prompt):
    prompt_template = f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {prompt}

    ### Response:
    '''

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1
    )

    return pipe(prompt_template)[0]['generated_text'].split('Response:\n')[-1]

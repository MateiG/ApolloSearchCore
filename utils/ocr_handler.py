import os
import re
from tqdm import tqdm

from pdf2image import convert_from_path
import pytesseract


class OCRHandler:
    UPLOAD_DIR = 'static/uploads/'

    def process_pdf(self, file_id):
        print('ocr processing pdf')
        self.chunk_id = 0
        images = convert_from_path(os.path.join(OCRHandler.UPLOAD_DIR, file_id + '.pdf'))

        chunks = []
        for page_num, image in tqdm(enumerate(images), total=len(images)):
            image = image.convert('L')
            page_width = image.width
            page_height = image.height

            pt_data = pytesseract.image_to_data(image, output_type='dict')
            pt_cleaned = self.filter_pt(pt_data)
            pt_extracted = self.extract_pt(pt_cleaned, page_width, page_height)
            pt_parsed = self.parse_pt(pt_extracted)
            chunks += self.create_chunks(pt_parsed, page_num + 1)
        return chunks

    def filter_pt(self, data):
        indices_to_keep = []
        for i in range(len(data['level'])):
            if (data['level'][i] == 5 and data['text'][i].strip() != ''):
                indices_to_keep.append(i)

        cleaned = {key: [value[i] for i in indices_to_keep] for key, value in data.items()}
        return cleaned

    def extract_pt(self, data, page_width, page_height):
        extracted = {}  # block, par, line
        for i in range(len(data['text'])): # for each token
            block = data['block_num'][i]
            par = data['par_num'][i]
            line = data['line_num'][i]
            text = data['text'][i]
            left = data['left'][i]
            top = data['top'][i]
            width = data['width'][i]
            height = data['height'][i]

            if (block not in extracted):
                extracted[block] = {}
            if (par not in extracted[block]):
                extracted[block][par] = {}
            if (line not in extracted[block][par]):
                extracted[block][par][line] = {
                    'text': [],
                    'left': [],
                    'top': [],
                    'width': [],
                    'height': []
                }

            extracted[block][par][line]['text'].append(text)
            extracted[block][par][line]['left'].append(left / page_width)
            extracted[block][par][line]['top'].append(top / page_height)
            extracted[block][par][line]['width'].append(width / page_width)
            extracted[block][par][line]['height'].append(height / page_height)
        return extracted
    
    def parse_pt(self, data):
        parsed = dict(data)
        for b_key in data:
            for p_key in data[b_key]:
                for l_key in data[b_key][p_key]:
                    num_tokens = len(data[b_key][p_key][l_key]['text'])
                    text = (' ').join(data[b_key][p_key][l_key]['text'])
                    left = min(data[b_key][p_key][l_key]['left'])
                    right = data[b_key][p_key][l_key]['left'][num_tokens - 1] + data[b_key][p_key][l_key]['width'][num_tokens - 1]
                    width = right - left
                    top = min(data[b_key][p_key][l_key]['top'])
                    height = max(data[b_key][p_key][l_key]['height'])
                    bottom = top + height

                    parsed[b_key][p_key][l_key]['text'] = text
                    parsed[b_key][p_key][l_key]['left'] = left
                    parsed[b_key][p_key][l_key]['right'] = right
                    parsed[b_key][p_key][l_key]['top'] = top
                    parsed[b_key][p_key][l_key]['width'] = width
                    parsed[b_key][p_key][l_key]['height'] = height
                    parsed[b_key][p_key][l_key]['bottom'] = bottom
        return parsed

    def create_chunks(self, data, page_num, chunk_size=3, min_chars=10):
        chunks = []
        for b_key in data:
            for p_key in data[b_key]:
                p_lines = list(data[b_key][p_key].values())
                for i in range(0, len(p_lines), chunk_size):
                    chunk_lines = p_lines[i:i+chunk_size]
                    num_lines = len(chunk_lines)
                    text = ('\n').join([line['text'] for line in chunk_lines])

                    left = min([line['left'] for line in chunk_lines])
                    right = max([line['right'] for line in chunk_lines])
                    width = right - left
                    top = min([line['top'] for line in chunk_lines])
                    height = chunk_lines[num_lines - 1]['top'] + chunk_lines[num_lines - 1]['height'] - top

                    if (len(text) >= min_chars):
                        chunks.append({
                            'id': self.chunk_id,
                            'page': page_num,
                            'text': text,
                            'box': [left, top, width, height]
                        })
                        self.chunk_id += 1
        return chunks

    def clean_string(self, text):
        pattern = r'[^a-zA-Z0-9!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~\s]'

        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(pattern, '', text)
        return text

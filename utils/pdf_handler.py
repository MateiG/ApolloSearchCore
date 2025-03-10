import os
import re

from pypdf import PdfReader

from nltk.tokenize import sent_tokenize, word_tokenize

from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.cloud import storage
from google.cloud import documentai_toolbox

PROJECT_ID = "apollosearch"
LOCATION = "us"
PROCESSOR_ID = "11f4f90dcc5e62d1"
GCS_INPUT_URI = "gs://apollo-search-storage/input/"
GCS_OUTPUT_URI = "gs://apollo-search-storage/output/"
MIME_TYPE = "application/pdf"


class PDFHandler:
    UPLOAD_DIR = "static/uploads/"

    def __init__(self):
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket("apollo-search-storage")

        opts = ClientOptions(api_endpoint="us-documentai.googleapis.com")
        self.docai_client = documentai.DocumentProcessorServiceClient(
            client_options=opts
        )

        self.RESOURCE_NAME = self.docai_client.processor_path(
            PROJECT_ID, LOCATION, PROCESSOR_ID
        )

    def clean_text(self, text):
        pattern = r'[^a-zA-Z0-9!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~\s]'

        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(pattern, "", text)
        return text

    def get_bounding_box(self, norm_verts):
        vertices = []
        for vert in norm_verts:
            vertices.append([vert.x, vert.y])

        x_values = [v[0] for v in vertices]
        y_values = [v[1] for v in vertices]
        min_x = min(x_values)
        max_x = max(x_values)
        min_y = min(y_values)
        max_y = max(y_values)

        width = max_x - min_x
        height = max_y - min_y

        return [min_x, min_y, width, height]

    def parse_docai_document(self, document, metadata):
        paragraph_id = 0
        documents = []
        all_text = document.text

        for page in document.pages:
            page = page.documentai_object
            for paragraph in page.blocks:
                par_seg_start = paragraph.layout.text_anchor.text_segments[
                    0
                ].start_index
                par_seg_end = paragraph.layout.text_anchor.text_segments[0].end_index
                par_text = all_text[par_seg_start:par_seg_end]

                par_text_chunks = self.chunk_text(par_text)
                for par_text_chunk in par_text_chunks:
                    box = self.get_bounding_box(
                        paragraph.layout.bounding_poly.normalized_vertices
                    )
                    parsed_doc = {
                        "id": paragraph_id,
                        "page": page.page_number,
                        "text": par_text_chunk,
                        "box": box,
                    }
                    documents.append(parsed_doc)
                    paragraph_id += 1
        return documents

    def online_process(self, file_id):
        print("online processing pdf")
        meta_reader = PdfReader(os.path.join(self.UPLOAD_DIR, file_id + ".pdf"))

        with open(os.path.join(PDFHandler.UPLOAD_DIR, file_id + ".pdf"), "rb") as image:
            image_content = image.read()
        raw_document = documentai.RawDocument(
            content=image_content, mime_type=MIME_TYPE
        )
        request = documentai.ProcessRequest(
            name=self.RESOURCE_NAME, raw_document=raw_document
        )
        result = self.docai_client.process_document(request=request)
        wrapped_document = (
            documentai_toolbox.document.Document.from_documentai_document(
                result.document
            )
        )
        documents = self.parse_docai_document(
            wrapped_document,
            metadata=(
                meta_reader.pages[0].mediabox.width,
                meta_reader.pages[0].mediabox.height,
            ),
        )
        print("finished online processing pdf")
        return documents

    def offline_process(self, file_id):
        print("offline processing pdf")
        meta_reader = PdfReader(os.path.join(self.UPLOAD_DIR, file_id + ".pdf"))

        blob = self.bucket.blob("input/" + file_id + ".pdf")
        blob.upload_from_filename(os.path.join(self.UPLOAD_DIR, file_id + ".pdf"))

        gcs_document = documentai.GcsDocument(
            gcs_uri=GCS_INPUT_URI + file_id + ".pdf", mime_type=MIME_TYPE
        )
        gcs_documents = documentai.GcsDocuments(documents=[gcs_document])
        gcs_output_config = documentai.DocumentOutputConfig.GcsOutputConfig(
            gcs_uri=GCS_OUTPUT_URI
        )
        input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)
        output_config = documentai.DocumentOutputConfig(
            gcs_output_config=gcs_output_config
        )

        request = documentai.BatchProcessRequest(
            name=self.RESOURCE_NAME,
            input_documents=input_config,
            document_output_config=output_config,
            skip_human_review=True,
        )
        operation = self.docai_client.batch_process_documents(request)
        operation.result()

        docai_document = (
            documentai_toolbox.document.Document.from_batch_process_operation(
                location=LOCATION, operation_name=operation.operation.name
            )[0]
        )
        parsed_documents = self.parse_docai_document(
            docai_document,
            metadata=(
                meta_reader.pages[0].mediabox.width,
                meta_reader.pages[0].mediabox.height,
            ),
        )
        print("finished offline processing pdf")
        return parsed_documents

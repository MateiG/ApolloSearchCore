import os
import re

from pypdf import PdfReader

from nltk.tokenize import sent_tokenize, word_tokenize

from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.cloud import storage
from google.cloud import documentai_toolbox


PROJECT_ID = 'apollosearch'
LOCATION = 'us'
PROCESSOR_ID = '11f4f90dcc5e62d1'
GCS_INPUT_URI = 'gs://apollo-search-storage/input/'
GCS_OUTPUT_URI = 'gs://apollo-search-storage/output/'
MIME_TYPE = 'application/pdf'


class PDFHandler():
    UPLOAD_DIR = 'static/uploads/'

    def __init__(self):
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket('apollo-search-storage')

        opts = ClientOptions(api_endpoint='us-documentai.googleapis.com')
        self.docai_client = documentai.DocumentProcessorServiceClient(client_options=opts)

        self.RESOURCE_NAME = self.docai_client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)

    def clean_string(self, text):
        pattern = r'[^a-zA-Z0-9!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~\s]'

        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(pattern, '', text)
        return text
        
    def split_text(self, text, chunk_size=3, min_words=3):
        text = self.clean_string(text)

        if (len(word_tokenize(text)) < min_words):
            return []

        # sentences = sent_tokenize(text)
        # long_sentences = []
        # for sent in sentences:
        #     if (len(word_tokenize(sent)) > min_words):
        #         long_sentences.append(sent)

        # num_sentences = len(sentences)
        # chunked_sentences = []
        # for i in range(0, num_sentences, chunk_size):
        #     chunk = sentences[i:i+chunk_size]
        #     chunk = ' '.join(chunk)
        #     word_count = len(word_tokenize(chunk))
        #     if (word_count >= min_words):
        #         chunked_sentences.append(chunk)
        return [text]

    def get_vertices(self, norm_verts, metadata):
        vertices = []
        for vert in norm_verts:
            vertices.append([int(vert.x * metadata[0]), int(vert.y * metadata[1])])
        return vertices

    def parse_docai_document(self, document, metadata):
        paragraph_id = 0
        documents = []

        for page in document.pages:
            for paragraph in page.paragraphs:
                chunks = self.split_text(paragraph.text)
                # block_chunks = self.split_block(block.text)
                for chunk in chunks:
                    vertices = self.get_vertices(paragraph.documentai_paragraph.layout.bounding_poly.normalized_vertices, metadata)
                    parsed_doc = {'id': paragraph_id, 'page': page.documentai_page.page_number,
                                  'text': chunk, 'vertices': vertices}
                    documents.append(parsed_doc)
                    paragraph_id += 1
        return documents

    def online_process(self, file_id):
        meta_reader = PdfReader(os.path.join(self.UPLOAD_DIR, file_id + '.pdf'))

        with open(os.path.join(PDFHandler.UPLOAD_DIR, file_id + '.pdf'), 'rb') as image:
            image_content = image.read()
        raw_document = documentai.RawDocument(content=image_content, mime_type=MIME_TYPE)
        request = documentai.ProcessRequest(name=self.RESOURCE_NAME, raw_document=raw_document)
        result = self.docai_client.process_document(request=request)
        wrapped_document = documentai_toolbox.document.Document.from_documentai_document(result.document)
        documents = self.parse_docai_document(wrapped_document, metadata=(
            meta_reader.pages[0].mediabox.width, meta_reader.pages[0].mediabox.height))
        return documents

    def offline_process(self, file_id):
        meta_reader = PdfReader(os.path.join(self.UPLOAD_DIR, file_id + '.pdf'))

        blob = self.bucket.blob('input/' + file_id + '.pdf')
        blob.upload_from_filename(os.path.join(self.UPLOAD_DIR, file_id + '.pdf'))

        gcs_document = documentai.GcsDocument(
            gcs_uri=GCS_INPUT_URI + file_id + '.pdf', mime_type=MIME_TYPE
        )
        gcs_documents = documentai.GcsDocuments(documents=[gcs_document])
        gcs_output_config = documentai.DocumentOutputConfig.GcsOutputConfig(
            gcs_uri=GCS_OUTPUT_URI
        )
        input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)
        output_config = documentai.DocumentOutputConfig(gcs_output_config=gcs_output_config)

        request = documentai.BatchProcessRequest(
            name=self.RESOURCE_NAME,
            input_documents=input_config,
            document_output_config=output_config,
            skip_human_review=True
        )
        operation = self.docai_client.batch_process_documents(request)
        operation.result()

        docai_document = documentai_toolbox.document.Document.from_batch_process_operation(
            location=LOCATION, operation_name=operation.operation.name
        )[0]
        parsed_documents = self.parse_docai_document(docai_document, metadata=(
            meta_reader.pages[0].mediabox.width, meta_reader.pages[0].mediabox.height))
        return parsed_documents

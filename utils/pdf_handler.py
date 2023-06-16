import os
import re

from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.cloud import storage

PROJECT_ID = 'apollosearch'
LOCATION = 'us'
PROCESSOR_ID = '11f4f90dcc5e62d1'
GCS_INPUT_URI = 'gs://apollo-search-storage/input/'
GCS_OUTPUT_URI = 'gs://apollo-search-storage/output/'
MIME_TYPE = 'application/pdf'


class PDFHandler():
    UPLOAD_DIR='static/uploads/'

    def __init__(self):
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket('apollo-search-storage')

        opts = ClientOptions(api_endpoint='us-documentai.googleapis.com')
        self.docai_client = documentai.DocumentProcessorServiceClient(client_options=opts)

        self.RESOURCE_NAME = self.docai_client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)

    def parse_document(self, document, existing_documents=[]):
        paragraph_id = 0
        documents = existing_documents

        text = document.text
        for page in document.pages:
            for paragraph in page.paragraphs:
                paragraph_text = ''
                for segment in paragraph.layout.text_anchor.text_segments:
                    start_index = int(segment.start_index)
                    end_index = int(segment.end_index)
                    paragraph_text += text[start_index:end_index]
                
                paragraph_text = re.sub(r'\s+', ' ', paragraph_text)
                if (len(paragraph_text) > 50):
                    parsed_doc = {'id': paragraph_id, 'page': page.page_number, 'text': paragraph_text}
                    documents.append(parsed_doc)
                    paragraph_id += 1
        return documents

    def online_process(self, file_id):
        with open(os.path.join(PDFHandler.UPLOAD_DIR, file_id + '.pdf'), 'rb') as image:
            image_content = image.read()
        raw_document = documentai.RawDocument(content=image_content, mime_type=MIME_TYPE)
        request = documentai.ProcessRequest(name=self.RESOURCE_NAME, raw_document=raw_document)
        result = self.docai_client.process_document(request=request)
        documents = self.parse_document(result.document)
        return documents
    
    def offline_process(self, file_id):
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

        metadata = documentai.BatchProcessMetadata(operation.metadata)
        process = metadata.individual_process_statuses[0]
        matches = re.match(r"gs://(.*?)/(.*)", process.output_gcs_destination)
        output_bucket, output_prefix = matches.groups()
        output_blobs = self.storage_client.list_blobs(output_bucket, prefix=output_prefix)

        documents = []
        for blob in output_blobs:
            document = documentai.Document.from_json(blob.download_as_bytes(), ignore_unknown_fields=True)
            documents = self.parse_document(document, documents)
        return documents


# handler = PDFHandler()
# output = handler.offline_process('1de4d540-737d-4aa1-9c57-862510ec5141')

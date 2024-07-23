from dotenv import load_dotenv,dotenv_values
import json
import os
from sentence_transformers import SentenceTransformer

from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient, SearchIndexingBufferedSender  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryCaptionResult,
    QueryAnswerResult,
    SemanticErrorMode,
    SemanticErrorReason,
    SemanticSearchResultsType,
    QueryType,
    VectorizedQuery,
    VectorQuery,
    VectorFilterMode,    
)
from azure.search.documents.indexes.models import (  
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    SemanticPrioritizedFields,
    SemanticField,  
    SearchField,  
    SemanticSearch,
    VectorSearch,  
    HnswAlgorithmConfiguration,
    HnswParameters,  
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchProfile,
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    ExhaustiveKnnParameters,
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    SemanticField,  
    SearchField,  
    VectorSearch,  
    HnswParameters,  
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)  
  
# Configure environment variables  
load_dotenv()  


class CustomSearchAI():

    def __init__(self):

        self.service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT") 
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME") 
        self.key = os.getenv("AZURE_SEARCH_ADMIN_KEY") 
        self.model = os.getenv("MODEL_NAME")
        self.credential = AzureKeyCredential(self.key)

        print("[+] credentials are set up")

        self.index_client = SearchIndexClient(endpoint=self.service_endpoint, credential=self.credential)
        self.fields = [ 
                        SimpleField(name="id", type=SearchFieldDataType.String,key=True, sortable=True, filterable=True, facetable=True),
                        SearchableField(name="line", type=SearchFieldDataType.String),
                        SearchableField(name="filename", type=SearchFieldDataType.String,filterable=True, facetable=True),
                        SearchableField(name="page_number", type=SearchFieldDataType.Int64,filterable=True, facetable=True),
                        SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),searchable=True, vector_search_dimensions=os.getenv("DIMENSION"), vector_search_profile_name="myHnswProfile")
                    ]
        
        self.vector_search_client = VectorSearch(
                                                    algorithms=[
                                                        HnswAlgorithmConfiguration(
                                                            name="myHnsw",
                                                            kind=VectorSearchAlgorithmKind.HNSW,
                                                            parameters=HnswParameters(
                                                                m=4,
                                                                ef_construction=400,
                                                                ef_search=500,
                                                                metric=VectorSearchAlgorithmMetric.COSINE
                                                            )
                                                        ),
                                                        ExhaustiveKnnAlgorithmConfiguration(
                                                            name="myExhaustiveKnn",
                                                            kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                                                            parameters=ExhaustiveKnnParameters(
                                                                metric=VectorSearchAlgorithmMetric.COSINE
                                                            )
                                                        )
                                                    ],
                                                    profiles=[
                                                        VectorSearchProfile(
                                                            name="myHnswProfile",
                                                            algorithm_configuration_name="myHnsw",
                                                        ),
                                                        VectorSearchProfile(
                                                            name="myExhaustiveKnnProfile",
                                                            algorithm_configuration_name="myExhaustiveKnn",
                                                        )
                                                    ]
                                                )
        print("[+] Vector Search Client is set up!")
        
        # self.semantic_config = SemanticConfiguration(
        #                                                 name="my-semantic-config",
        #                                                 prioritized_fields=SemanticPrioritizedFields(
        #                                                     content_fields=[SemanticField(field_name="line")],
        #                                                     keywords_fields=[SemanticField(field_name="filename")]
        #                                                 )
        #                                             )
        
        # Create the semantic settings with the configuration
        # self.semantic_search = SemanticSearch(configurations=[self.semantic_config])

        # Create the search index with the semantic settings
        self.index = SearchIndex(name=self.index_name, fields=self.fields,
                            vector_search=self.vector_search_client)
        
        self.index_result = self.index_client.create_or_update_index(self.index)


        print("[+] Search Client is set up!")

    def upload_docs(self, docs):
        search_client = SearchClient(endpoint=self.service_endpoint, index_name=self.index_name, credential=self.credential)
        result = search_client.upload_documents(docs)
        return result
    







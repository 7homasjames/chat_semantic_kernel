from fastapi import FastAPI
from dotenv import load_dotenv,dotenv_values
import json
import os
from sentence_transformers import SentenceTransformer

from pydantic import BaseModel
from typing import List, Optional
from openai import AzureOpenAI

from semantic_kernel.functions import KernelArguments
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings, OpenAITextEmbedding
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
    AzureChatPromptExecutionSettings
)


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

from sentence_transformers import SentenceTransformer
  
# Configure environment variables  
load_dotenv()  

service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT") 
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME") 
key = os.getenv("AZURE_SEARCH_ADMIN_KEY") 
model = os.getenv("MODEL_NAME")
credential = AzureKeyCredential(key)

index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)
fields = [ 
                SimpleField(name="id", type=SearchFieldDataType.String,key=True, sortable=True, filterable=True, facetable=True),
                SearchableField(name="line", type=SearchFieldDataType.String),
                SearchableField(name="filename", type=SearchFieldDataType.String,filterable=True, facetable=True),
                SearchableField(name="page_number", type=SearchFieldDataType.String,filterable=True, facetable=True),
                SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),searchable=True, vector_search_dimensions=os.getenv("DIMENSION"), vector_search_profile_name="myHnswProfile")
            ]

vector_search_client = VectorSearch(
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


index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search_client)
index_result = index_client.create_or_update_index(index)

search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)

model = SentenceTransformer(os.getenv("MODEL_NAME"))

##########################################################################################
# Initialize Semantic Kernel
kernel = Kernel()

# Prompt Template for Chat Completion with Grounding
prompt_template = """
    You are a chatbot that can have a conversation about any topic related to the provided context.
    Give explicit answers from the provided context or say 'I don't know' if it does not have an answer.
    Provided context: {{$db_record}}

    User: {{$query_term}}
    Chatbot:"""

if os.getenv("GLOBAL_LLM_SERVICE") == "OpenAI":

    # Add OpenAI Chat Completion Service
    openai_service = OpenAIChatCompletion(
        api_key=os.getenv("OPENAI_API_KEY"),
        ai_model_id="gpt-3.5-turbo"
    )
    kernel.add_service(openai_service)

    chat_execution_settings = OpenAIChatPromptExecutionSettings(
        ai_model_id="gpt-3.5-turbo",
        max_tokens=1000,
        temperature=0.0,
        top_p=0.5
    )

else:

    azure_openai_service = AzureChatCompletion(
            service_id="chat_completion",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        )
    kernel.add_service(azure_openai_service)

    chat_execution_settings = AzureChatPromptExecutionSettings(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            max_tokens=1000,
            temperature=0.0,
            top_p=0.5
        )

chat_prompt_template_config = PromptTemplateConfig(
    template=prompt_template,
    name="grounded_response",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="db_record", description="The database record", is_required=True),
        InputVariable(name="query_term", description="The user input", is_required=True),
    ],
    execution_settings=chat_execution_settings,
)

chat_function = kernel.add_function(
        function_name="ChatGPTFunc",
        plugin_name="chatGPTPlugin",
        prompt_template_config=chat_prompt_template_config
        )


##########################################################################################

app = FastAPI()

#########################################################################################

class Item(BaseModel):
    id: str
    line: str
    filename: str
    page_number: str
    #embedding: List[float]

class Docs(BaseModel):
    items: List[Item]

class Query(BaseModel):
    query: str
    #page_number: Optional[int] 
    
    # embedding: List[float]

class QA(BaseModel):
    query: str
    context: str

#################################################################################################

@app.post("/push_docs/")
async def push_docs(item: Docs):
    print(item.model_dump(mode="json")["items"])

    try:
        docs = item.model_dump(mode="json")["items"]
        for doc in docs:
            doc['embedding'] = model.encode(doc['line']).reshape(-1).astype(float).tolist()

        result = search_client.upload_documents(docs)
        return result
    except Exception as e:
        print(e)



@app.post("/context/")
async def context(item: Query):

    try:
    
        query = item.model_dump(mode="json")["query"]
        #page_number = item.model_dump(mode="json")["page_number"]
        
        query_vector = model.encode(query).reshape(-1).astype(float).tolist()
        vector_query = VectorizedQuery(vector=query_vector, 
                               k_nearest_neighbors=3, 
                               fields="embedding")

        results = search_client.search(  
                                        search_text=None,  
                                        vector_queries=[vector_query],
                                        select=["line", "filename","page_number"],
                                        top=5
                                    )
        
        resp = {"docs" : [], 
                "filenames" : [],
                "page_number": [],
                "paragraph_numbers": []}
        
        # Collect the results

        for idx, result in enumerate(results):
            print("Results;",results)
            resp["docs"].append(result['line'])
            resp["filenames"].append(result['filename'])
            resp["page_number"].append(result['page_number'])
        return resp
    
    except Exception as e:
        print(e)


@app.post("/response/")
async def response(item: QA):

    try:
        query = item.model_dump(mode="json")["query"]
        context = item.model_dump(mode="json")["context"]
        arguments = KernelArguments(db_record=context, query_term=query)

        result = await kernel.invoke(
            chat_function,arguments
        )

        print(result)

        return {"output" : f"{result}" }
    
    except Exception as e:
        print(e)


    


    
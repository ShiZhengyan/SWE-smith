from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, ChainedTokenCredential, AzureCliCredential, get_bearer_token_provider
import re
import os

ep = "o3"
# ep = "o3-mini"
# ep = "o4-mini"
# ep = "4o"
# Add new models
# ep = "gpt-4.1"
# ep = "gpt-4.5-preview"
# ep = "o1"
# ep = "gpt-4.1-mini"

scope = "api://trapi/.default"
credential = get_bearer_token_provider(ChainedTokenCredential(
    AzureCliCredential(),
    DefaultAzureCredential(
        exclude_cli_credential=True,
        # Exclude other credentials we are not interested in.
        exclude_environment_credential=True,
        exclude_shared_token_cache_credential=True,
        exclude_developer_cli_credential=True,
        exclude_powershell_credential=True,
        exclude_interactive_browser_credential=True,
        exclude_visual_studio_code_credentials=True,
        # DEFAULT_IDENTITY_CLIENT_ID is a variable exposed in
        # Azure ML Compute jobs that has the client id of the
        # user-assigned managed identity in it.
        # See https://learn.microsoft.com/en-us/azure/machine-learning/how-to-identity-based-service-authentication#compute-cluster
        # In case it is not set the ManagedIdentityCredential will
        # default to using the system-assigned managed identity, if any.
        managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
    )
),scope)

if ep == "4o":
    model_name = 'gpt-4o'  # Ensure this is a valid model name
    model_version = '2024-05-13'  # Ensure this is a valid model version
    instance = 'gcr/preview' # See https://aka.ms/trapi/models for the instance name, remove /openai (library adds it implicitly)     
    api_version = '2024-10-21'  # Ensure this is a valid API version see: https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release
elif ep == "o3":
    model_name = 'o3'  # Ensure this is a valid model name
    model_version = '2025-04-16'  # Ensure this is a valid model version
    instance = 'msrne/shared' # See https://aka.ms/trapi/models for the instance name, remove /openai (library adds it implicitly) 
    api_version = '2025-04-01-preview'
elif ep == "o3-mini":
    model_name = 'o3-mini'  # Ensure this is a valid model name
    model_version = '2025-01-31'  # Ensure this is a valid model version
    instance = 'msrne/shared' # See https://aka.ms/trapi/models for the instance name, remove /openai (library adds it implicitly) 
    api_version = '2025-04-01-preview'
elif ep == "o4-mini":
    model_name = 'o4-mini'  # Ensure this is a valid model name
    model_version = '2025-04-16'  # Ensure this is a valid model version
    instance = 'msrne/shared' # See https://aka.ms/trapi/models for the instance name, remove /openai (library adds it implicitly) 
    api_version = '2025-04-01-preview'
elif ep == "gpt-4.1":
    model_name = 'gpt-4.1'  # Ensure this is a valid model name
    model_version = '2025-04-14'  # Ensure this is a valid model version
    instance = 'gcr/shared' # See https://aka.ms/trapi/models for the instance name, remove /openai (library adds it implicitly) 
    api_version = '2025-04-01-preview'
elif ep == "gpt-4.5-preview":
    model_name = 'gpt-4.5-preview'  # Ensure this is a valid model name
    model_version = '2025-02-27'  # Ensure this is a valid model version
    instance = 'msrne/shared' # See https://aka.ms/trapi/models for the instance name, remove /openai (library adds it implicitly) 
    api_version = '2025-04-01-preview'
elif ep == "o1":
    model_name = 'o1'  # Ensure this is a valid model name
    model_version = '2024-12-17'  # Ensure this is a valid model version
    instance = 'msrne/shared' # See https://aka.ms/trapi/models for the instance name, remove /openai (library adds it implicitly) 
    api_version = '2025-04-01-preview'
elif ep == "gpt-4.1-mini":
    model_name = 'gpt-4.1-mini'
    model_version = '2025-04-14'  # Ensure this is a valid model version
    instance = 'msrne/shared'
    api_version = '2025-04-01-preview'

deployment_name = re.sub(r"[^a-zA-Z0-9._-]", "", f"{model_name}_{model_version}")
print(f'Using model: {model_name}, version: {model_version}, deployment name: {deployment_name}')
endpoint = f'https://trapi.research.microsoft.com/{instance}'

client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_ad_token_provider=credential,
    api_version=api_version,
)
# for m in client.models.list().data:
#     print(m.id)
tool_list = [
    {
        "type": "function",
        "function": {
            "name": "get_captal",
            "description": "get the capital city of a country",
            # "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": ["string", "null"],
                        "description": "returns the capital city name."
                    }
                },
                "required": ["country"],
                "additionalProperties": False
            }
        }
    },
]


response = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "Give a one word answer, what is the capital of France?",
        },
    ],
    tools=tool_list,
    tool_choice="required",
)
response_content = response.choices[0].message
print(response_content)


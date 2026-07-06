// =============================================================================
// Glass-Box Video Analyst — production infrastructure (Azure).
//
// Topology:
//   - API  (FastAPI, replay-only)   -> Azure Container Apps, image pulled from ACR.
//   - Web  (Next.js static export)  -> Azure Static Web Apps (content pushed by the workflow).
//   - No Blob / Key Vault: the API replays committed cache JSON and its run-cache is ephemeral;
//     the only runtime secret is the Azure OpenAI key (live free-text codegen).
//
// First-deploy ordering: the API container app starts on a PUBLIC placeholder image (apiImage
// default) so provisioning never blocks on an image that hasn't been pushed yet. The deploy
// workflow then builds+pushes the real image to ACR and `az containerapp update`s the app to it.
// The app's user-assigned identity holds AcrPull, so that later pull just works.
// =============================================================================

@description('Azure region for the Container Apps + ACR resources.')
param location string = resourceGroup().location

@description('Region for the Static Web App (must be a SWA-supported region).')
@allowed([ 'eastus2', 'centralus', 'westus2', 'eastasia', 'westeurope' ])
param swaLocation string = 'eastus2'

@description('Short prefix for resource names (lowercase letters/digits only).')
@minLength(3)
@maxLength(11)
param namePrefix string = 'glassbox'

@description('API container image. Defaults to a public placeholder for first provision; the deploy workflow updates it to the ACR image.')
param apiImage string = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'

@description('Comma-separated production web origins allowed by the API CORS (e.g. the SWA URL). Usually set by the workflow after the SWA hostname is known.')
param allowedOrigins string = ''

@description('Keep at least one API replica warm (1 = snappy first impression, 0 = scale-to-zero to save cost).')
@minValue(0)
@maxValue(5)
param apiMinReplicas int = 1

@description('Azure OpenAI endpoint for live codegen. Blank disables live free-text (canned + pinned still work).')
param azureOpenAiEndpoint string = ''

@description('Azure OpenAI API key.')
@secure()
param azureOpenAiKey string = ''

@description('Azure OpenAI gpt-4o deployment name.')
param azureOpenAiDeployment string = 'gpt-4o'

@description('Azure OpenAI API version.')
param azureOpenAiApiVersion string = '2024-10-21'

// ---- derived names -----------------------------------------------------------
var acrName = toLower('${namePrefix}acr${uniqueString(resourceGroup().id)}')
var envName = '${namePrefix}-env'
var apiAppName = '${namePrefix}-api'
var swaName = '${namePrefix}-web'
var uamiName = '${namePrefix}-api-id'
var logName = '${namePrefix}-logs'
var acrPullRoleId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')

// ---- observability -----------------------------------------------------------
resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logName
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// ---- container registry ------------------------------------------------------
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false } // pull via managed identity, no admin creds
}

// ---- identity for credential-less ACR pull -----------------------------------
resource uami 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: uamiName
  location: location
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, uami.id, acrPullRoleId)
  scope: acr
  properties: {
    principalId: uami.properties.principalId
    roleDefinitionId: acrPullRoleId
    principalType: 'ServicePrincipal'
  }
}

// ---- container apps environment ----------------------------------------------
resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: envName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

// ---- API container app -------------------------------------------------------
resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: apiAppName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${uami.id}': {} }
  }
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto' // HTTP/1.1 + h2; SSE (the /api/run stream) rides HTTP/1.1
        allowInsecure: false
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: uami.id
        }
      ]
      secrets: [
        {
          name: 'azure-openai-key'
          value: azureOpenAiKey
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          resources: {
            cpu: json('0.5')
            memory: '1.0Gi'
          }
          env: [
            { name: 'ALLOWED_ORIGINS', value: allowedOrigins }
            { name: 'AZURE_OPENAI_ENDPOINT', value: azureOpenAiEndpoint }
            { name: 'AZURE_OPENAI_KEY', secretRef: 'azure-openai-key' }
            { name: 'AZURE_OPENAI_DEPLOYMENT', value: azureOpenAiDeployment }
            { name: 'AZURE_OPENAI_API_VERSION', value: azureOpenAiApiVersion }
          ]
        }
      ]
      scale: {
        minReplicas: apiMinReplicas
        maxReplicas: 2
      }
    }
  }
  dependsOn: [ acrPull ] // ensure the pull role exists before the (later) ACR image rolls in
}

// ---- web static site ---------------------------------------------------------
resource swa 'Microsoft.Web/staticSites@2023-12-01' = {
  name: swaName
  location: swaLocation
  sku: { name: 'Free', tier: 'Free' }
  properties: {} // content is pushed by the deploy workflow (SWA CLI), not a linked repo build
}

// ---- outputs (consumed by the deploy workflow) -------------------------------
output acrName string = acr.name
output acrLoginServer string = acr.properties.loginServer
output apiAppName string = api.name
output apiFqdn string = api.properties.configuration.ingress.fqdn
output apiUrl string = 'https://${api.properties.configuration.ingress.fqdn}'
output swaName string = swa.name
output swaHostname string = swa.properties.defaultHostname
output swaUrl string = 'https://${swa.properties.defaultHostname}'

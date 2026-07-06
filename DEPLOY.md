# Deploying the Glass-Box Video Analyst to Azure

Production deploy is **infrastructure-as-code (Bicep) + a gated GitHub Actions workflow**.
One human does the one-time setup below; after that, every deploy is a single click of the
**deploy** workflow.

## Architecture

| Component | Hosts | Notes |
|-----------|-------|-------|
| **API** (`api/`, FastAPI) | Azure **Container Apps** | Replay-only at runtime. Image built from `api/Dockerfile`, pulled from ACR via a managed identity. The only runtime secret is the Azure OpenAI key (live free-text codegen). |
| **Web** (`web/`, Next.js static export) | Azure **Static Web Apps** | `next build` → `web/out/`. `NEXT_PUBLIC_API_BASE` is baked in at build time to the API's public URL, so the workflow builds web *after* the API exists. |

No Blob storage or Key Vault: the API replays committed cache JSON, its run-cache is ephemeral,
and the single secret lives as a Container App secret. **Azure Vision creds are NOT needed in
production** — vision runs only in the offline `scripts/precompute.py` pipeline.

> The first deploy starts the API on a public placeholder image, then the workflow rolls it to
> the freshly-built ACR image and sets `ALLOWED_ORIGINS` to the Static Web App's URL. Re-runs are
> non-disruptive: the provision step reads the live image + CORS and passes them back through, so
> the running app is never reset to the placeholder — the new image rolls in as one extra revision.
>
> On a **brand-new subscription**, register the resource providers once during setup (step 1
> below). The RG-scoped deploy principal can't self-register them (that's a subscription-level
> action), so a human with subscription rights does it once up front.

---

## One-time setup (human, needs your Azure subscription)

Prereqs: `az` CLI logged in (`az login`), `gh` CLI logged in, Owner on the subscription (or
rights to create a resource group + a role assignment).

### 1. Resource group + resource providers

```bash
az group create --name glass-box-rg --location eastus2

# One-time per subscription: register the resource providers the deploy uses. A brand-new
# subscription hasn't registered these, and the RG-scoped deploy principal (step 2) can't do it
# itself, so register them here with your subscription rights. Fast no-op if already registered.
for ns in Microsoft.ContainerRegistry Microsoft.App Microsoft.Web \
          Microsoft.OperationalInsights Microsoft.ManagedIdentity; do
  az provider register --namespace "$ns" --wait
done
```

### 2. GitHub OIDC identity (no stored client secret)

```bash
SUB=$(az account show --query id -o tsv)
TENANT=$(az account show --query tenantId -o tsv)
OWNER_REPO="<your-gh-owner>/<your-gh-repo>"   # e.g. laielli/video-analyst-app

# App registration + service principal
az ad app create --display-name "glass-box-deploy"
APP_ID=$(az ad app list --display-name "glass-box-deploy" --query "[0].appId" -o tsv)
az ad sp create --id "$APP_ID"

# Federated credential. The deploy job uses `environment: production`, so the OIDC SUBJECT is
# the environment form (NOT the branch ref). This exact subject is what makes the login work.
az ad app federated-credential create --id "$APP_ID" --parameters "{
  \"name\": \"github-prod-env\",
  \"issuer\": \"https://token.actions.githubusercontent.com\",
  \"subject\": \"repo:${OWNER_REPO}:environment:production\",
  \"audiences\": [\"api://AzureADTokenExchange\"]
}"

# The SP needs deploy rights AND the ability to assign the one AcrPull role the Bicep creates.
# Least privilege = Contributor + User Access Administrator, scoped to the RG (recommended).
# (Grant a single "Owner" instead only if you prefer one command over two — it's broader.)
az role assignment create --assignee "$APP_ID" --role "Contributor" \
  --scope "/subscriptions/${SUB}/resourceGroups/glass-box-rg"
az role assignment create --assignee "$APP_ID" --role "User Access Administrator" \
  --scope "/subscriptions/${SUB}/resourceGroups/glass-box-rg"
```

### 3. GitHub Actions secrets + variables

```bash
# Secrets (sensitive)
gh secret set AZURE_CLIENT_ID        --body "$APP_ID"
gh secret set AZURE_TENANT_ID        --body "$TENANT"
gh secret set AZURE_SUBSCRIPTION_ID  --body "$SUB"
gh secret set AZURE_OPENAI_ENDPOINT  --body "https://<your-resource>.openai.azure.com/"  # your gpt-4o endpoint
gh secret set AZURE_OPENAI_KEY       --body "<your-azure-openai-key>"

# Variables (non-sensitive config)
gh variable set AZURE_RESOURCE_GROUP   --body "glass-box-rg"
gh variable set NAME_PREFIX            --body "glassbox"
gh variable set AZURE_OPENAI_DEPLOYMENT --body "gpt-4o"
```

> Leave `AZURE_OPENAI_*` blank to ship a deploy where live free-text codegen is disabled —
> the canned catalog + pinned-hero replays still work end to end.

### 4. GitHub environment

Create a `production` environment on the repo (Settings → Environments) — optionally add a
required reviewer so deploys need approval. The federated-credential subject above already
targets this environment.

---

## Deploy

```bash
gh workflow run deploy.yml
gh run watch
```

The workflow: provisions Bicep → cloud-builds + pushes the API image to ACR → rolls the
Container App to it and sets CORS to the web origin → builds the static web bundle with the API
URL baked in → uploads it to Static Web Apps. The run summary prints both public URLs.

## Verify

```bash
API=$(az containerapp show -n glassbox-api -g glass-box-rg --query properties.configuration.ingress.fqdn -o tsv)
curl -s "https://$API/api/catalog" | head        # API reachable
WEB=$(az staticwebapp show -n glassbox-web -g glass-box-rg --query defaultHostname -o tsv)
echo "open https://$WEB"                          # gallery front door
```

Open the web URL, run a catalog query, and confirm the live step stream renders (that exercises
the cross-origin SSE path the `ALLOWED_ORIGINS` CORS step enables).

## Cost / teardown

- Container Apps `minReplicas: 1` keeps the API warm (snappier first impression). Set the Bicep
  `apiMinReplicas: 0` param to scale to zero between visits and cut idle cost.
- Static Web Apps Free tier + Basic ACR + a single small Container App is a low monthly footprint.
- Tear everything down with `az group delete --name glass-box-rg --yes --no-wait`.

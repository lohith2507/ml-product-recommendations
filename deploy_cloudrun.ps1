# Deploy the shopping agent to Cloud Run on project shopping-agent-513607.
# Prerequisites:
#   1. gcloud auth application-default login (lohithdattavarmaveepuri@gmail.com)
#   2. python setup_vertex_index.py
#   3. Docker Desktop running (or use Cloud Build)

$ErrorActionPreference = "Stop"
$Project = "shopping-agent-513607"
$Region = "us-central1"          # Cloud Run + Artifact Registry + embeddings
$ChatLocation = "global"         # Gemini 3.x chat
$Service = "shopping-agent"
$Image = "$Region-docker.pkg.dev/$Project/shopping-agent/$Service`:latest"

gcloud config set project $Project
gcloud config set account lohithdattavarmaveepuri@gmail.com

if (-not (Test-Path "products_vertex.index")) {
    Write-Host "Building Vertex FAISS index..."
    $env:GOOGLE_CLOUD_PROJECT = $Project
    $env:GOOGLE_CLOUD_LOCATION = $Region
    python setup_vertex_index.py
}

gcloud artifacts repositories describe shopping-agent --location=$Region 2>$null
if ($LASTEXITCODE -ne 0) {
    gcloud artifacts repositories create shopping-agent `
        --repository-format=docker `
        --location=$Region `
        --description="Shopping agent images"
}

gcloud auth configure-docker "$Region-docker.pkg.dev" --quiet

Write-Host "Building image $Image ..."
docker build -t $Image .

Write-Host "Pushing..."
docker push $Image

Write-Host "Deploying Cloud Run service..."
gcloud run deploy $Service `
    --image=$Image `
    --region=$Region `
    --platform=managed `
    --allow-unauthenticated `
    --memory=2Gi `
    --cpu=2 `
    --timeout=300 `
    --set-env-vars="GOOGLE_CLOUD_PROJECT=$Project,GOOGLE_CLOUD_LOCATION=$ChatLocation,VERTEX_CHAT_LOCATION=$ChatLocation,VERTEX_EMBED_LOCATION=$Region,VERTEX_CHAT_MODEL=gemini-3.5-flash,VERTEX_FALLBACK_MODEL=gemini-3.1-flash-lite,USE_VERTEX=1" `
    --service-account="$Service@$Project.iam.gserviceaccount.com"

gcloud run services describe $Service --region=$Region --format="value(status.url)"

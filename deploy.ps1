$IMAGE = "us-central1-docker.pkg.dev/us-energy-data/dash-apps/us-datacentres:latest"

Write-Host "Building and pushing image..." -ForegroundColor Cyan
gcloud builds submit --tag $IMAGE
if ($LASTEXITCODE -ne 0) { Write-Host "Build failed." -ForegroundColor Red; exit 1 }

Write-Host "Deploying to Cloud Run..." -ForegroundColor Cyan
gcloud run deploy us-datacentres --image $IMAGE --region us-central1 --platform managed --allow-unauthenticated --memory 1Gi --cpu 1
if ($LASTEXITCODE -ne 0) { Write-Host "Deploy failed." -ForegroundColor Red; exit 1 }

Write-Host "Done. App is live at https://datacentres.eliswyn.com" -ForegroundColor Green

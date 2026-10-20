$ErrorActionPreference = 'Stop'

$demoHost = 'mia-platform'
$certificatePassword = 'mia-platform-local-demo'
$projectRoot = Split-Path -Parent $PSScriptRoot
$certificateDirectory = Join-Path $projectRoot '.cert'
$pfxPath = Join-Path $certificateDirectory "$demoHost.pfx"
$cerPath = Join-Path $certificateDirectory "$demoHost.cer"
$hostsPath = Join-Path $env:SystemRoot 'System32\drivers\etc\hosts'
$hostsEntry = "127.0.0.1 $demoHost"

if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  throw 'Run PowerShell as Administrator, then run npm run demo:setup from the frontend directory. Administrator access is required to update the Windows hosts file.'
}

New-Item -ItemType Directory -Force -Path $certificateDirectory | Out-Null

if (-not (Select-String -Path $hostsPath -Pattern "^\s*127\.0\.0\.1\s+$([regex]::Escape($demoHost))(\s|$)" -Quiet)) {
  Add-Content -Path $hostsPath -Value "`r`n# MIA Platform local demo`r`n$hostsEntry"
  Write-Host "Added hosts mapping: $hostsEntry"
} else {
  Write-Host "Hosts mapping already exists: $hostsEntry"
}

if (-not (Test-Path $pfxPath)) {
  $certificate = New-SelfSignedCertificate `
    -DnsName $demoHost, 'localhost' `
    -CertStoreLocation 'Cert:\CurrentUser\My' `
    -KeyAlgorithm RSA `
    -KeyLength 2048 `
    -HashAlgorithm SHA256 `
    -KeyExportPolicy Exportable `
    -NotAfter (Get-Date).AddYears(3)

  $securePassword = ConvertTo-SecureString -String $certificatePassword -AsPlainText -Force
  Export-PfxCertificate -Cert $certificate -FilePath $pfxPath -Password $securePassword | Out-Null
  Export-Certificate -Cert $certificate -FilePath $cerPath | Out-Null
  Import-Certificate -FilePath $cerPath -CertStoreLocation 'Cert:\CurrentUser\Root' | Out-Null
  Write-Host "Created and trusted the local HTTPS certificate for $demoHost."
} else {
  Write-Host 'Local HTTPS certificate already exists.'
}

Write-Host ''
Write-Host 'Setup complete. Start with: npm.cmd run demo'
Write-Host "Browser address: https://$demoHost"

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('scienceqa', 'mmqa', 'vqax')]
    [string]$Dataset
)

$ErrorActionPreference = 'Stop'
function Invoke-Python([string[]]$Arguments) {
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $($Arguments -join ' ')" }
}

Push-Location $PSScriptRoot
try {
    switch ($Dataset) {
        'scienceqa' {
            Invoke-Python -Arguments @('download_scienceqa.py')
            Invoke-Python -Arguments @('prepare_scienceqa_mira_stages.py', '--kg-model', 'gpt-5',
                            '--output', 'outputs/scienceqa_full_stages')
            Invoke-Python -Arguments @('train_mira_scienceqa.py', '--prepared', 'outputs/scienceqa_full_stages',
                            '--decoder-mode', 'attention', '--compression',
                            '--generate-explanations', '--output', 'outputs/scienceqa_full')
        }
        'mmqa' {
            Invoke-Python -Arguments @('download_mmqa.py', '--skip-image-archive')
            Invoke-Python -Arguments @('prepare_mmqa_dataset.py')
            Invoke-Python -Arguments @('extract_mmqa_images_remote.py')
            Invoke-Python -Arguments @('prepare_openqa_mira_stages.py', '--dataset', 'mmqa',
                            '--output', 'outputs/mmqa_full_stages')
            Invoke-Python -Arguments @('train_mira_openqa.py', '--dataset', 'mmqa',
                            '--prepared', 'outputs/mmqa_full_stages', '--compression',
                            '--output', 'outputs/mmqa_full')
        }
        'vqax' {
            Invoke-Python -Arguments @('prepare_vqax_dataset.py')
            Invoke-Python -Arguments @('prepare_openqa_mira_stages.py', '--dataset', 'vqax',
                            '--output', 'outputs/vqax_full_stages')
            Invoke-Python -Arguments @('train_mira_openqa.py', '--dataset', 'vqax',
                            '--prepared', 'outputs/vqax_full_stages', '--compression',
                            '--output', 'outputs/vqax_full')
        }
    }
} finally {
    Pop-Location
}

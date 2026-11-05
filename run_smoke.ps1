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
            Invoke-Python -Arguments @('prepare_scienceqa_mira_stages.py', '--train-count', '10',
                            '--test-count', '5', '--kg-model', 'gpt-5',
                            '--output', 'outputs/scienceqa_smoke_stages')
            Invoke-Python -Arguments @('train_mira_scienceqa.py', '--train-count', '10', '--test-count', '5',
                            '--epochs', '1', '--prepared', 'outputs/scienceqa_smoke_stages',
                            '--decoder-mode', 'attention', '--compression',
                            '--generate-explanations', '--output', 'outputs/scienceqa_smoke')
        }
        'mmqa' {
            Invoke-Python -Arguments @('download_mmqa.py', '--skip-image-archive')
            Invoke-Python -Arguments @('prepare_mmqa_dataset.py', '--train-count', '10', '--val-count', '5')
            Invoke-Python -Arguments @('extract_mmqa_images_remote.py')
            Invoke-Python -Arguments @('prepare_openqa_mira_stages.py', '--dataset', 'mmqa',
                            '--output', 'outputs/mmqa_smoke_stages')
            Invoke-Python -Arguments @('train_mira_openqa.py', '--dataset', 'mmqa', '--train-limit', '10',
                            '--val-limit', '5', '--prepared', 'outputs/mmqa_smoke_stages',
                            '--compression', '--output', 'outputs/mmqa_smoke')
        }
        'vqax' {
            Invoke-Python -Arguments @('prepare_vqax_dataset.py', '--train-count', '10', '--val-count', '5')
            Invoke-Python -Arguments @('prepare_openqa_mira_stages.py', '--dataset', 'vqax',
                            '--output', 'outputs/vqax_smoke_stages')
            Invoke-Python -Arguments @('train_mira_openqa.py', '--dataset', 'vqax', '--train-limit', '10',
                            '--val-limit', '5', '--prepared', 'outputs/vqax_smoke_stages',
                            '--compression', '--output', 'outputs/vqax_smoke')
        }
    }
} finally {
    Pop-Location
}

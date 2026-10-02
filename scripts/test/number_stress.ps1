param([string]$Root = (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent))
# Historical failed SAPI attempt: voices were unavailable in this execution
# security context. Use the separately frozen number_stress.py v2 generator.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$planPath = Join-Path $Root 'experiments/vcm-complete-v1/number-stress-plan.json'
$plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
$receiptPath = Join-Path $Root 'experiments/vcm-complete-v1/number-stress-manifest.jsonl'
if (Test-Path -LiteralPath $receiptPath) { throw 'Stress manifest exists; preserve it and use a new version.' }
$synthesizer = New-Object System.Speech.Synthesis.SpeechSynthesizer
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$outputRoot = [System.IO.Path]::GetFullPath((Join-Path $Root '.cache/vcm-complete-v1/number-stress/audio'))
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$writer = [System.IO.StreamWriter]::new($receiptPath, $false, [System.Text.UTF8Encoding]::new($false))
try {
    $count = 0
    foreach ($item in $plan.rows) {
        $path = [System.IO.Path]::GetFullPath((Join-Path $Root $item.audio))
        if (-not $path.StartsWith($outputRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Audio outside declared cache.' }
        if (Test-Path -LiteralPath $path) { throw ('Refusing to overwrite ' + $path) }
        $synthesizer.SelectVoice($item.voice)
        $synthesizer.SetOutputToWaveFile($path, $format)
        $synthesizer.Speak($item.transcript)
        $synthesizer.SetOutputToNull()
        $item | Add-Member -NotePropertyName audio_sha256 -NotePropertyValue (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        $writer.WriteLine(($item | ConvertTo-Json -Depth 8 -Compress))
        $writer.Flush()
        $count++
        if ($count % 50 -eq 0) { Write-Output ('SYNTHETIC_STRESS ' + $count + '/' + $plan.rows.Count) }
    }
} finally {
    $writer.Dispose()
    $synthesizer.Dispose()
}
Write-Output ('SYNTHETIC_STRESS_COMPLETE ' + $count)

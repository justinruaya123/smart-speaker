param(
    [string]$Prompts = "experiments/pcv-human-slots-v1/synthesis-prompts.json",
    [string]$Output = ".cache/pcv-human-slots-v1/sapi"
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$pcvDestination = [System.IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $pcvDestination) { throw 'Choose a new generation directory; existing evidence is immutable.' }
[System.IO.Directory]::CreateDirectory($pcvDestination) | Out-Null
$pcvRows = Get-Content -Raw -LiteralPath $Prompts | ConvertFrom-Json
$pcvVoices = @(
    @{ Name = 'Microsoft David'; Split = 'train'; Id = 'david' },
    @{ Name = 'Microsoft Mark'; Split = 'dev'; Id = 'mark' },
    @{ Name = 'Microsoft Zira'; Split = 'test'; Id = 'zira' }
)
$pcvSynth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$pcvFormat = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$pcvManifest = New-Object System.IO.StreamWriter((Join-Path $pcvDestination 'manifest.jsonl'), $false, (New-Object System.Text.UTF8Encoding($false)))
try {
    $pcvVoiceInfo = @()
    foreach ($pcvVoice in $pcvVoices) {
        $pcvSynth.SelectVoice($pcvVoice.Name)
        $pcvSynth.Rate = 0
        $pcvVoiceInfo += @{ name = $pcvSynth.Voice.Name; id = $pcvSynth.Voice.Id; culture = $pcvSynth.Voice.Culture.Name; version = $pcvSynth.Voice.AdditionalInfo['Version']; split = $pcvVoice.Split }
        $pcvVoiceDir = Join-Path $pcvDestination $pcvVoice.Id
        [System.IO.Directory]::CreateDirectory($pcvVoiceDir) | Out-Null
        foreach ($pcvRow in $pcvRows) {
            $pcvAudio = Join-Path $pcvVoiceDir ($pcvRow.id + '.wav')
            $pcvText = if ($pcvRow.spoken_text) { $pcvRow.spoken_text } else { $pcvRow.text }
            $pcvSynth.SetOutputToWaveFile($pcvAudio, $pcvFormat)
            $pcvSynth.Speak($pcvText)
            $pcvSynth.SetOutputToNull()
            $pcvRecord = @{
                id = ('sapi/' + $pcvVoice.Id + '/' + $pcvRow.id); phrase_id = $pcvRow.id;
                source = 'SAPI'; engine = 'Windows System.Speech'; speaker = ('sapi/' + $pcvVoice.Id);
                pair = ('sapi/' + $pcvVoice.Id + '/' + $pcvRow.id); split = $pcvVoice.Split;
                audio = $pcvAudio; expected_audio_sha256 = (Get-FileHash -LiteralPath $pcvAudio -Algorithm SHA256).Hash.ToLower();
                text = $pcvRow.text; spoken_text = $pcvText; command = $pcvRow.command;
                synthetic = $true; human_speaker = $false; tier = $pcvRow.tier;
                rate = 0; spoken_text_listener_verified = $false
            }
            $pcvManifest.WriteLine(($pcvRecord | ConvertTo-Json -Compress -Depth 12))
        }
        $pcvManifest.Flush()
        Write-Output ('Completed synthetic voice ' + $pcvVoice.Name + ': ' + $pcvRows.Count)
    }
    @{
        generated_utc = [DateTime]::UtcNow.ToString('o'); voices = $pcvVoiceInfo;
        os = [System.Environment]::OSVersion.VersionString; runtime = [System.Environment]::Version.ToString();
        prompt_sha256 = (Get-FileHash -LiteralPath $Prompts -Algorithm SHA256).Hash.ToLower();
        source_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLower();
        provenance = 'Installed Microsoft English voices; local research synthesis; no human enrollment or external service';
        limitation = 'One development and one test voice of one engine; not cross-engine qualification; no independent listening'
    } | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $pcvDestination 'generation.json') -Encoding utf8
} finally {
    $pcvManifest.Dispose()
    $pcvSynth.Dispose()
}

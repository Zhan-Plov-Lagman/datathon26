[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$DataDir,
    [string]$OutputPath = '.\artifacts\dataset_audit.generated.json'
)

$ErrorActionPreference = 'Stop'
$dataRoot = (Resolve-Path -LiteralPath $DataDir).Path
$labelPath = Join-Path $dataRoot 'open_labels.xlsx'
$videoRoot = Join-Path $dataRoot 'videos'
$outputFull = [System.IO.Path]::GetFullPath($OutputPath)
$artifactRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\artifacts'))
$artifactPrefix = $artifactRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
if (-not $outputFull.StartsWith($artifactPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'OutputPath must be inside this repository artifacts directory.'
}
if (Test-Path -LiteralPath $outputFull) { throw 'Output already exists. Choose a new OutputPath.' }
if (-not (Test-Path -LiteralPath $labelPath -PathType Leaf)) { throw 'Missing open_labels.xlsx.' }
if (-not (Test-Path -LiteralPath $videoRoot -PathType Container)) { throw 'Missing videos directory.' }
$probe = (Get-Command ffprobe -ErrorAction Stop).Source
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($labelPath)
$issues = [System.Collections.Generic.List[string]]::new()
try {
    $sharedStrings = @()
    $sharedEntry = $archive.GetEntry('xl/sharedStrings.xml')
    if ($null -ne $sharedEntry) {
        $reader = [System.IO.StreamReader]::new($sharedEntry.Open())
        try { [xml]$sharedXml = $reader.ReadToEnd() } finally { $reader.Dispose() }
        $ns = [System.Xml.XmlNamespaceManager]::new($sharedXml.NameTable)
        $ns.AddNamespace('s', 'http://schemas.openxmlformats.org/spreadsheetml/2006/main')
        $sharedStrings = @($sharedXml.SelectNodes('//s:si', $ns) | ForEach-Object {
            ($_.SelectNodes('.//s:t', $ns) | ForEach-Object { $_.InnerText }) -join ''
        })
    }
    $entry = $archive.GetEntry('xl/worksheets/sheet1.xml')
    if ($null -eq $entry) { throw 'Expected first worksheet is missing.' }
    $reader = [System.IO.StreamReader]::new($entry.Open())
    try { [xml]$sheet = $reader.ReadToEnd() } finally { $reader.Dispose() }
    $ns = [System.Xml.XmlNamespaceManager]::new($sheet.NameTable)
    $ns.AddNamespace('s', 'http://schemas.openxmlformats.org/spreadsheetml/2006/main')
    $labels = @($sheet.SelectNodes('//s:sheetData/s:row', $ns) | Select-Object -Skip 1 | ForEach-Object {
        $cells = @{}
        foreach ($cell in $_.SelectNodes('s:c', $ns)) {
            if ($null -ne $cell.SelectSingleNode('s:f', $ns)) { throw 'Formula cells require explicit review.' }
            $column = $cell.r -replace '\d', ''
            if ($cell.t -eq 'inlineStr') {
                $cells[$column] = ($cell.SelectNodes('.//s:t', $ns) | ForEach-Object { $_.InnerText }) -join ''
            } elseif ($cell.t -eq 's') {
                $cells[$column] = $sharedStrings[[int]$cell.SelectSingleNode('s:v', $ns).InnerText]
            } else {
                $value = $cell.SelectSingleNode('s:v', $ns)
                if ($null -ne $value) { $cells[$column] = $value.InnerText }
            }
        }
        if (-not [string]::IsNullOrWhiteSpace($cells['A'])) {
            $raw = $cells['B']
            $seconds = $null
            $numbers = [regex]::Matches([string]$raw, '\d+')
            if ($numbers.Count -eq 1) { $seconds = [int]$numbers[0].Value }
            elseif ($numbers.Count -eq 2 -and [int]$numbers[1].Value -lt 60) {
                $seconds = 60 * [int]$numbers[0].Value + [int]$numbers[1].Value
            } elseif (-not [string]::IsNullOrWhiteSpace($raw)) {
                $issues.Add("Unparsed timestamp: $($cells['A'])")
            }
            [pscustomobject]@{name=$cells['A'];label=$cells['C'];timestamp_raw=$raw;timestamp_sec=$seconds}
        }
    })
} finally { $archive.Dispose() }

$files = @(Get-ChildItem -LiteralPath $videoRoot -File | Where-Object Extension -ieq '.mov')
if ($files.Count -eq 0) { throw 'No MOV files found.' }
$videos = @($files | ForEach-Object {
    $file = $_
    $metadataText = & $probe -v error -show_entries 'format=duration,size:stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate:stream_side_data=rotation' -of json $file.FullName
    if ($LASTEXITCODE -ne 0) { throw "ffprobe failed: $($file.Name)" }
    $metadata = ($metadataText | Out-String) | ConvertFrom-Json
    $duration = [double]::Parse($metadata.format.duration, [System.Globalization.CultureInfo]::InvariantCulture)
    $matchesForFile = @($labels | Where-Object { $_.name -ceq $file.BaseName })
    if ($matchesForFile.Count -ne 1) { $issues.Add("Label match count $($matchesForFile.Count): $($file.Name)") }
    $matchedLabel = $null
    if ($matchesForFile.Count -eq 1) { $matchedLabel = $matchesForFile[0] }
    if ($null -ne $matchedLabel.timestamp_sec -and ($matchedLabel.timestamp_sec -lt 0 -or $matchedLabel.timestamp_sec -gt $duration)) {
        $issues.Add("Timestamp outside video: $($file.Name)")
    }
    [pscustomobject]@{
        file=$file.Name;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        duration_sec=$duration;streams=@($metadata.streams);label_matches=$matchesForFile.Count
        label=$matchedLabel.label;timestamp_raw=$matchedLabel.timestamp_raw;timestamp_sec=$matchedLabel.timestamp_sec
    }
})
foreach ($label in $labels) {
    if (@($files | Where-Object { $_.BaseName -ceq $label.name }).Count -ne 1) {
        $issues.Add("Video match is not unique: $($label.name)")
    }
}
$groups = @($videos | Group-Object sha256)
$duplicates = @($groups | Where-Object Count -gt 1 | ForEach-Object {
    [pscustomobject]@{files=@($_.Group.file);labels=@($_.Group.label);timestamps=@($_.Group.timestamp_sec);sha256=$_.Name}
})
$report = [pscustomobject]@{
    generated_at_utc=[DateTime]::UtcNow.ToString('o');label_rows=$labels.Count;video_count=$videos.Count
    unique_sha256=$groups.Count;total_bytes=($videos | Measure-Object bytes -Sum).Sum
    total_duration_sec=($videos | Measure-Object duration_sec -Sum).Sum
    duration_min=($videos | Measure-Object duration_sec -Minimum).Minimum
    duration_max=($videos | Measure-Object duration_sec -Maximum).Maximum
    label_counts=@($videos | Group-Object label | Select-Object Name,Count)
    duplicates=$duplicates;issues=@($issues.ToArray());videos=$videos
}
[System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($outputFull)) | Out-Null
[System.IO.File]::WriteAllText($outputFull, ($report | ConvertTo-Json -Depth 12), [System.Text.UTF8Encoding]::new($false))
Write-Output "Videos: $($videos.Count); labels: $($labels.Count); unique hashes: $($groups.Count); duplicate groups: $($duplicates.Count); issues: $($issues.Count)"
Write-Output "Local report: $outputFull"
if ($issues.Count -gt 0) { Write-Warning 'Review issues in the local report before using the dataset.' }

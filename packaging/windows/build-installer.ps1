# Build dist\HushMic-Setup-<version>.exe from the current tree.
# Requires Inno Setup 6 (winget install JRSoftware.InnoSetup) and the
# MSVC host used to compile hushmic-win. Models, ONNX Runtime, and the
# OpenVINO NPU runtime are taken from the local provisioning folders.

param(
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Stage = Join-Path $Root "dist\installer-stage"
$Dist = Join-Path $Root "dist"
$Icon = Join-Path $Dist "hushmic.ico"

function Get-AppVersion {
    $cargo = Get-Content (Join-Path $Root "Cargo.toml") -Raw
    if ($cargo -notmatch '(?m)^version\s*=\s*"([^"]+)"') {
        throw "version not found in Cargo.toml"
    }
    return $Matches[1]
}

function Invoke-ReleaseBuild {
    $cargo = Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"
    if (-not (Test-Path $cargo)) {
        throw "cargo not found at $cargo"
    }
    $vcvars = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path $vswhere) {
            $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
            if ($vs) {
                $vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
            }
        }
    }
    if (-not (Test-Path $vcvars)) {
        throw "vcvars64.bat not found. Install the MSVC C++ build tools."
    }
    Write-Host "Building hushmic-win (release)..."
    cmd /c "call `"$vcvars`" >nul && `"$cargo`" build --release -p hushmic-win"
    if ($LASTEXITCODE -ne 0) {
        throw "cargo build --release -p hushmic-win failed ($LASTEXITCODE)"
    }
}

function New-PngIcon([string]$PngPath, [string]$IcoPath) {
    $png = [System.IO.File]::ReadAllBytes($PngPath)
    $ms = New-Object System.IO.MemoryStream
    $bw = New-Object System.IO.BinaryWriter($ms)
    $bw.Write([UInt16]0)
    $bw.Write([UInt16]1)
    $bw.Write([UInt16]1)
    $bw.Write([byte]0) # 256
    $bw.Write([byte]0) # 256
    $bw.Write([byte]0)
    $bw.Write([byte]0)
    $bw.Write([UInt16]1)
    $bw.Write([UInt16]32)
    $bw.Write([UInt32]$png.Length)
    $bw.Write([UInt32]22)
    $bw.Write($png)
    $bw.Flush()
    [System.IO.File]::WriteAllBytes($IcoPath, $ms.ToArray())
    $bw.Close()
}

function Copy-License([string]$Source, [string]$DestName) {
    if (Test-Path $Source) {
        Copy-Item $Source (Join-Path $Stage "licenses\$DestName")
    }
}

function Find-VcRedist {
    $candidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022\BuildTools\VC\Redist\MSVC\v143\vc_redist.x64.exe")
    )
    $found = Get-ChildItem (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio") -Recurse -Filter "vc_redist.x64.exe" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch "\\debug\\" } |
        Sort-Object FullName -Descending |
        Select-Object -First 1
    if ($found) { return $found.FullName }
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

function Find-Iscc {
    $candidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    throw "ISCC.exe not found. Install Inno Setup 6: winget install JRSoftware.InnoSetup"
}

$Version = Get-AppVersion
if (-not $SkipBuild) {
    Invoke-ReleaseBuild
}

$Exe = Join-Path $Root "target\release\hushmic-win.exe"
$Ort = Join-Path $Root "target\release\onnxruntime.dll"
if (-not (Test-Path $Ort)) {
    $Ort = Join-Path $Root ".build-tools\onnxruntime\onnxruntime-win-x64-1.27.0\lib\onnxruntime.dll"
}
$OvRoot = Get-ChildItem (Join-Path $Root ".build-tools") -Directory -Filter "openvino_toolkit_windows_*" |
    Sort-Object Name -Descending |
    Select-Object -First 1
if (-not $OvRoot) {
    throw "OpenVINO toolkit not found under .build-tools"
}
$OvBin = Join-Path $OvRoot.FullName "runtime\bin\intel64\Release"
$TbbBin = Join-Path $OvRoot.FullName "runtime\3rdparty\tbb\bin"

$required = @(
    $Exe,
    $Ort,
    (Join-Path $OvBin "openvino.dll"),
    (Join-Path $OvBin "openvino_c.dll"),
    (Join-Path $OvBin "openvino_intel_npu_plugin.dll"),
    (Join-Path $OvBin "openvino_onnx_frontend.dll"),
    (Join-Path $OvBin "openvino_ir_frontend.dll"),
    (Join-Path $TbbBin "tbb12.dll"),
    (Join-Path $TbbBin "tbbmalloc.dll"),
    (Join-Path $Root "assets\models\dpdfnet8_48khz_hr.onnx"),
    (Join-Path $Root "assets\models\dpdfnet2_48khz_hr.onnx"),
    (Join-Path $Root "assets\models\dpdfnet8_48khz_hr.front.onnx"),
    (Join-Path $Root "assets\models\dpdfnet8_48khz_hr.body.onnx"),
    (Join-Path $Root "assets\models\dpdfnet8_48khz_hr.split.txt"),
    (Join-Path $Root "assets\models\dpdfnet2_48khz_hr.front.onnx"),
    (Join-Path $Root "assets\models\dpdfnet2_48khz_hr.body.onnx"),
    (Join-Path $Root "assets\models\dpdfnet2_48khz_hr.split.txt")
)
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "missing installer input: $path"
    }
}

if (Test-Path $Stage) {
    Remove-Item $Stage -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $Stage, (Join-Path $Stage "models"), (Join-Path $Stage "licenses"), $Dist | Out-Null

Copy-Item $Exe (Join-Path $Stage "hushmic-win.exe")
Copy-Item $Ort (Join-Path $Stage "onnxruntime.dll")
foreach ($name in @(
        "openvino.dll",
        "openvino_c.dll",
        "openvino_intel_npu_plugin.dll",
        "openvino_onnx_frontend.dll",
        "openvino_ir_frontend.dll"
    )) {
    Copy-Item (Join-Path $OvBin $name) (Join-Path $Stage $name)
}
Copy-Item (Join-Path $TbbBin "tbb12.dll") (Join-Path $Stage "tbb12.dll")
Copy-Item (Join-Path $TbbBin "tbbmalloc.dll") (Join-Path $Stage "tbbmalloc.dll")
Copy-Item (Join-Path $Root "assets\models\*.onnx") (Join-Path $Stage "models")
Copy-Item (Join-Path $Root "assets\models\*.split.txt") (Join-Path $Stage "models")
Copy-License (Join-Path $Root "LICENSE-MIT") "LICENSE-MIT"
Copy-License (Join-Path $Root "LICENSE-APACHE") "LICENSE-APACHE"
Copy-License (Join-Path $Root ".build-tools\onnxruntime\onnxruntime-win-x64-1.27.0\LICENSE") "ONNXRuntime-LICENSE"
Copy-License (Join-Path $Root ".build-tools\onnxruntime\onnxruntime-win-x64-1.27.0\ThirdPartyNotices.txt") "ONNXRuntime-ThirdPartyNotices.txt"
Copy-License (Join-Path $OvRoot.FullName "docs\licensing\EULA.txt") "OpenVINO-EULA.txt"
Copy-License (Join-Path $OvRoot.FullName "docs\licensing\redist.txt") "OpenVINO-redist.txt"
Copy-License (Join-Path $OvRoot.FullName "docs\licensing\runtime-third-party-programs.txt") "OpenVINO-runtime-third-party-programs.txt"
Copy-License (Join-Path $OvRoot.FullName "docs\licensing\Apache_license.txt") "OpenVINO-Apache-LICENSE"
Copy-License (Join-Path $OvRoot.FullName "runtime\3rdparty\tbb\LICENSE") "TBB-LICENSE"
Get-ChildItem (Join-Path $OvRoot.FullName "runtime\3rdparty\tbb") -Filter "*LICENSE*" -File -ErrorAction SilentlyContinue |
    ForEach-Object { Copy-License $_.FullName $_.Name }

New-PngIcon (Join-Path $Root "packaging\hushmic-256.png") $Icon
Copy-Item $Icon (Join-Path $Stage "hushmic.ico")

Write-Host "Checking the staged layout (ONNX CPU, light model, 5 hops)..."
$onnxReport = (& (Join-Path $Stage "hushmic-win.exe") bench --engine onnx --model light --hops 5) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw "staged hushmic-win failed the ONNX bench:`n$onnxReport"
}
Write-Host $onnxReport

Write-Host "Checking the staged layout (OpenVINO NPU, light model, 5 hops)..."
$npuReport = (& (Join-Path $Stage "hushmic-win.exe") bench --engine npu --model light --hops 5) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw "staged hushmic-win failed the NPU bench:`n$npuReport"
}
Write-Host $npuReport
if ($npuReport -notmatch "active_engine: openvino npu") {
    throw "staged NPU runtime did not load.`n$npuReport"
}

$VcRedist = Find-VcRedist
$Iscc = Find-Iscc
$VcLine = ""
if ($VcRedist) {
    $VcEsc = $VcRedist.Replace("\", "\\")
    $VcLine = "#define VcRedist `"$VcEsc`"`r`n"
} else {
    Write-Warning "vc_redist.x64.exe not found. The setup will not install the Visual C++ runtime."
}

$StageEsc = $Stage.Replace("\", "\\")
$IconEsc = $Icon.Replace("\", "\\")
$config = @"
#define AppVersion "$Version"
#define StageDir "$StageEsc"
#define SetupIcon "$IconEsc"
$VcLine
"@
$utf8bom = New-Object System.Text.UTF8Encoding $true
[System.IO.File]::WriteAllText((Join-Path $PSScriptRoot "build-config.iss"), $config.Trim() + "`r`n", $utf8bom)

$iss = Join-Path $PSScriptRoot "hushmic.iss"
$issText = [System.IO.File]::ReadAllText($iss)
if (-not $issText.StartsWith([char]0xFEFF)) {
    [System.IO.File]::WriteAllText($iss, $issText, $utf8bom)
}

Write-Host "Compiling installer with $Iscc"
& $Iscc $iss
if ($LASTEXITCODE -ne 0) {
    throw "ISCC failed ($LASTEXITCODE)"
}

$setup = Join-Path $Dist "HushMic-Setup-$Version.exe"
if (-not (Test-Path $setup)) {
    throw "installer was not written to $setup"
}
$item = Get-Item $setup
Write-Host ("Installer: {0} ({1:N1} MB)" -f $item.FullName, ($item.Length / 1MB))

; HushMic Windows setup. Compile with packaging\windows\build-installer.ps1,
; which writes build-config.iss (version, staged payload, icon, VC++ redist).

#ifndef AppVersion
#include "build-config.iss"
#endif

#define AppName "HushMic"
#define AppExe "hushmic-win.exe"
#define AppPublisher "HushMic"
#define AppId "{{7B8E4A2C-1D5F-4E93-A6C0-8F2D4B7E91A3}"

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppVerName={#AppName} {#AppVersion}
VersionInfoVersion={#AppVersion}.0
VersionInfoProductName={#AppName}
DefaultDirName={autopf}\HushMic
DefaultGroupName=HushMic
DisableProgramGroupPage=yes
LicenseFile=..\..\LICENSE-MIT
OutputDir=..\..\dist
OutputBaseFilename=HushMic-Setup-{#AppVersion}
SetupIconFile={#SetupIcon}
UninstallDisplayIcon={app}\hushmic.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no
ChangesEnvironment=yes
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Messages]
english.WelcomeLabel2=This installs HushMic %1, real-time microphone noise suppression for Windows.%n%nThe Intel NPU path needs the Intel AI Boost driver. Without it, HushMic uses the CPU. Calls need VB-Audio Virtual Cable: pick CABLE Output as the microphone.
russian.WelcomeLabel2=Программа установит HushMic %1 — шумоподавление микрофона в реальном времени для Windows.%n%nРежим Intel NPU требует драйвер Intel AI Boost. Без него HushMic работает на процессоре. Для звонков нужен VB-Audio Virtual Cable: выберите CABLE Output как микрофон.
english.FinishedLabel=HushMic is installed.%n%nOpen a new terminal to use the hushmic-win command. For calls, install VB-Audio Virtual Cable and select CABLE Output as the microphone.
russian.FinishedLabel=HushMic установлен.%n%nОткройте новый терминал, чтобы команда hushmic-win была в PATH. Для звонков установите VB-Audio Virtual Cable и выберите CABLE Output как микрофон.

[Tasks]
Name: "addpath"; Description: "{cm:AddPath}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "autostart"; Description: "{cm:Autostart}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[CustomMessages]
english.AddPath=Add the hushmic-win command to PATH
russian.AddPath=Добавить команду hushmic-win в PATH
english.Autostart=Start HushMic when I sign in
russian.Autostart=Запускать HushMic при входе в Windows
english.VcRedistStatus=Installing the Visual C++ runtime...
russian.VcRedistStatus=Установка среды Visual C++...
english.AutostartStatus=Enabling start on login...
russian.AutostartStatus=Включение автозапуска...

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
#ifdef VcRedist
Source: "{#VcRedist}"; DestDir: "{tmp}"; Flags: deleteafterinstall
#endif

[Icons]
Name: "{group}\HushMic"; Filename: "{app}\{#AppExe}"; IconFilename: "{app}\hushmic.ico"
Name: "{group}\Uninstall HushMic"; Filename: "{uninstallexe}"
Name: "{autodesktop}\HushMic"; Filename: "{app}\{#AppExe}"; IconFilename: "{app}\hushmic.ico"; Tasks: desktopicon

[Run]
#ifdef VcRedist
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "{cm:VcRedistStatus}"; Check: VCRedistNeeded; Flags: waituntilterminated
#endif
Filename: "{app}\{#AppExe}"; Parameters: "config set autostart true"; StatusMsg: "{cm:AutostartStatus}"; Tasks: autostart; Flags: runhidden waituntilterminated
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
function VCRedistNeeded: Boolean;
var
  Installed: Cardinal;
begin
  if RegQueryDWordValue(HKLM64, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) then
    Result := Installed <> 1
  else
    Result := True;
end;

procedure EnvAddPath(PathToAdd: string);
var
  Paths: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Paths) then
    Paths := '';
  if Pos(';' + Uppercase(PathToAdd) + ';', ';' + Uppercase(Paths) + ';') > 0 then
    Exit;
  if Paths = '' then
    Paths := PathToAdd
  else if Copy(Paths, Length(Paths), 1) = ';' then
    Paths := Paths + PathToAdd
  else
    Paths := Paths + ';' + PathToAdd;
  RegWriteStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Paths);
end;

procedure EnvRemovePath(PathToRemove: string);
var
  Paths, Item, Rest, Rebuilt: string;
  P: Integer;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Paths) then
    Exit;
  Rest := Paths;
  Rebuilt := '';
  while Rest <> '' do
  begin
    P := Pos(';', Rest);
    if P = 0 then
    begin
      Item := Rest;
      Rest := '';
    end
    else
    begin
      Item := Copy(Rest, 1, P - 1);
      Rest := Copy(Rest, P + 1, MaxInt);
    end;
    if (Item <> '') and (CompareText(Item, PathToRemove) <> 0) then
    begin
      if Rebuilt <> '' then
        Rebuilt := Rebuilt + ';';
      Rebuilt := Rebuilt + Item;
    end;
  end;
  RegWriteStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Rebuilt);
end;

function SendMessageTimeout(hWnd: Integer; Msg: Cardinal; wParam: Longint; lParam: String;
  fuFlags, uTimeout: Cardinal; var lpdwResult: Cardinal): Integer;
  external 'SendMessageTimeoutW@user32.dll stdcall';

procedure RefreshEnvironment;
var
  Dummy: Cardinal;
begin
  Dummy := 0;
  SendMessageTimeout($FFFF, $001A, 0, 'Environment', 2, 5000, Dummy);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if WizardIsTaskSelected('addpath') then
    begin
      EnvAddPath(ExpandConstant('{app}'));
      RefreshEnvironment;
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ResultCode: Integer;
  RunValue: string;
begin
  if CurUninstallStep = usUninstall then
  begin
    Exec(ExpandConstant('{app}\{#AppExe}'), 'quit', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    Exec(ExpandConstant('{app}\{#AppExe}'), 'config set autostart false', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    EnvRemovePath(ExpandConstant('{app}'));
    RefreshEnvironment;
    if RegQueryStringValue(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Run', 'HushMic', RunValue) then
    begin
      if Pos(Lowercase(ExpandConstant('{app}')), Lowercase(RunValue)) > 0 then
        RegDeleteValue(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Run', 'HushMic');
    end;
  end;
end;

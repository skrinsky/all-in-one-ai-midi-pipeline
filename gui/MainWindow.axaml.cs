using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Threading;

namespace AiMidiGui;

public partial class MainWindow : Window
{
    private bool _isBusy;
    private bool _hasRunBatch = false;   // ✅ ADD THIS LINE
    public MainWindow()
    {
        InitializeComponent();

        // Actions
        RunBatchButton.Click += RunBatchButton_OnClick;
        ReviewPendingButton.Click += ReviewPendingButton_OnClick;
        ExportMidiButton.Click += ExportMidiButton_OnClick;
        ClearLogButton.Click += ClearLogButton_OnClick;

        // Browse buttons
        BrowsePythonButton.Click += BrowsePythonButton_OnClick;
        BrowseRepoButton.Click += BrowseRepoButton_OnClick;
        BrowseRawButton.Click += BrowseRawButton_OnClick;
        BrowseExportButton.Click += BrowseExportButton_OnClick;

        // ✅ You added this in XAML, so you MUST wire it too:
        BrowseSplitExportButton.Click += BrowseSplitExportButton_OnClick;
        ExportMidiButton.IsEnabled = false;

        StatusTextBlock.Text = "Idle";
    }

    // ---------------------- Button handlers ----------------------

    private void ClearLogButton_OnClick(object? sender, RoutedEventArgs e)
    {
        ClearLog();
    }

    private async void RunBatchButton_OnClick(object? sender, RoutedEventArgs e)
    {
        if (_isBusy) return;

        ClearLog();
        SetBusy("Running: run-batch …");

        string pythonExe = PythonPathTextBox.Text?.Trim() ?? "";
        string repoRoot = RepoRootTextBox.Text?.Trim() ?? "";
        string rawFolder = RawFolderTextBox.Text?.Trim() ?? "data/raw";
        bool normalizeKey = NormalizeKeyCheckBox.IsChecked ?? false;

        if (!File.Exists(pythonExe))
        {
            AppendLog($"ERROR: Python not found at {pythonExe}\n");
            ClearBusy("Idle (error)");
            return;
        }

        if (!Directory.Exists(repoRoot))
        {
            AppendLog($"ERROR: Repo root not found at {repoRoot}\n");
            ClearBusy("Idle (error)");
            return;
        }

        string glob = Path.Combine(rawFolder, "*.wav");
        string args = $"pipeline.py run-batch \"{glob}\"";

        if (normalizeKey)
            args += " --normalize-key";

        await RunProcessAsync(pythonExe, args, repoRoot, "run-batch");
        
        _hasRunBatch = true;
        ExportMidiButton.IsEnabled = true;
        AppendLog("\n✔ Batch complete — Export MIDI is now enabled.\n");
    }

    private async void ReviewPendingButton_OnClick(object? sender, RoutedEventArgs e)
    {
        if (_isBusy) return;

        ClearLog();
        SetBusy("Running: review-pending …");

        string pythonExe = PythonPathTextBox.Text?.Trim() ?? "";
        string repoRoot = RepoRootTextBox.Text?.Trim() ?? "";

        if (!File.Exists(pythonExe) || !Directory.Exists(repoRoot))
        {
            AppendLog("ERROR: Check Python path and repo root.\n");
            ClearBusy("Idle (error)");
            return;
        }

        string args = "pipeline.py review-pending";
        await RunProcessAsync(pythonExe, args, repoRoot, "review-pending");
    }

    private async void ExportMidiButton_OnClick(object? sender, RoutedEventArgs e)
    {
        if (!_hasRunBatch)
        {
            AppendLog("ERROR: Run Batch must be completed before exporting MIDI.\n");
            return;
        }
        
        if (_isBusy)
        {
            AppendLog("[WARN] Busy – export ignored.\n");
            return;
        }

        ClearLog();
        AppendLog("Export MIDI clicked.\n");
        SetBusy("Running: export-midi …");

        string pythonExe = PythonPathTextBox.Text?.Trim() ?? "";
        string repoRoot = RepoRootTextBox.Text?.Trim() ?? "";
        string exportFolder = ExportFolderTextBox.Text?.Trim() ?? "out_midis";

        if (!File.Exists(pythonExe))
        {
            AppendLog($"ERROR: Python not found: {pythonExe}\n");
            ClearBusy("Idle (error)");
            return;
        }

        if (!Directory.Exists(repoRoot))
        {
            AppendLog($"ERROR: Repo root not found: {repoRoot}\n");
            ClearBusy("Idle (error)");
            return;
        }

        // Export folder (absolute path for scanning, but CLI arg can be relative)
        string fullExportPath = Path.IsPathRooted(exportFolder)
            ? exportFolder
            : Path.Combine(repoRoot, exportFolder);

        Directory.CreateDirectory(fullExportPath);

        // 1) Export combined MIDIs
        string args = $"pipeline.py export-midi --out \"{exportFolder}\"";
        await RunProcessAsync(pythonExe, args, repoRoot, "export-midi");

        // If nothing exists, tell you clearly (otherwise it looks like "did nothing")
        var exported = Directory.EnumerateFiles(fullExportPath, "*.mid", SearchOption.TopDirectoryOnly).ToList();
        if (exported.Count == 0)
        {
            AppendLog($"[WARN] No .mid files found in export folder: {fullExportPath}\n");
            ClearBusy("Idle");
            return;
        }

        // 2) Optional: split each exported MIDI into per-track MIDIs
        bool doSplit = SplitTracksCheckBox.IsChecked ?? false;
        if (doSplit)
        {
            string splitFolder = SplitExportFolderTextBox.Text?.Trim() ?? "out_midis_split";

            string fullSplitPath = Path.IsPathRooted(splitFolder)
                ? splitFolder
                : Path.Combine(repoRoot, splitFolder);

            Directory.CreateDirectory(fullSplitPath);

            AppendLog("\n--- Splitting exported MIDIs into per-track files ---\n");
            await SplitAllExportedMidisAsync(pythonExe, repoRoot, fullExportPath, fullSplitPath);
        }

        ClearBusy("Idle");
    }




    // ---------------------- Browse buttons ----------------------

    private async void BrowsePythonButton_OnClick(object? sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Title = "Select Python executable",
            AllowMultiple = false
        };

        var result = await dlg.ShowAsync(this);
        if (result != null && result.Length > 0)
            PythonPathTextBox.Text = result[0];
    }

    private async void BrowseRepoButton_OnClick(object? sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog
        {
            Title = "Select AI MIDI repo root"
        };

        var result = await dlg.ShowAsync(this);
        if (!string.IsNullOrWhiteSpace(result))
            RepoRootTextBox.Text = result;
    }

    private async void BrowseRawButton_OnClick(object? sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog
        {
            Title = "Select raw audio folder"
        };

        var result = await dlg.ShowAsync(this);
        if (!string.IsNullOrWhiteSpace(result))
        {
            string repoRoot = RepoRootTextBox.Text?.Trim() ?? "";
            if (!string.IsNullOrWhiteSpace(repoRoot) &&
                result.StartsWith(repoRoot, StringComparison.Ordinal))
            {
                RawFolderTextBox.Text = result.Substring(repoRoot.Length).TrimStart(Path.DirectorySeparatorChar);
            }
            else
            {
                RawFolderTextBox.Text = result;
            }
        }
    }

    private async void BrowseExportButton_OnClick(object? sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog
        {
            Title = "Select export folder (or parent)"
        };

        var result = await dlg.ShowAsync(this);
        if (!string.IsNullOrWhiteSpace(result))
        {
            string repoRoot = RepoRootTextBox.Text?.Trim() ?? "";
            if (!string.IsNullOrWhiteSpace(repoRoot) &&
                result.StartsWith(repoRoot, StringComparison.Ordinal))
            {
                ExportFolderTextBox.Text =
                    result.Substring(repoRoot.Length).TrimStart(Path.DirectorySeparatorChar);
            }
            else
            {
                ExportFolderTextBox.Text = result;
            }
        }
    }

    // ✅ new browse for split folder
    private async void BrowseSplitExportButton_OnClick(object? sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog
        {
            Title = "Select split export folder (or parent)"
        };

        var result = await dlg.ShowAsync(this);
        if (!string.IsNullOrWhiteSpace(result))
        {
            string repoRoot = RepoRootTextBox.Text?.Trim() ?? "";
            if (!string.IsNullOrWhiteSpace(repoRoot) &&
                result.StartsWith(repoRoot, StringComparison.Ordinal))
            {
                SplitExportFolderTextBox.Text =
                    result.Substring(repoRoot.Length).TrimStart(Path.DirectorySeparatorChar);
            }
            else
            {
                SplitExportFolderTextBox.Text = result;
            }
        }
    }



    // ---------------------- Process runner ----------------------

    private async Task RunProcessAsync(string fileName, string arguments, string workingDirectory, string label)
    {
        AppendLog($"> {fileName} {arguments}\n\n");

        var psi = new ProcessStartInfo
        {
            FileName = fileName,
            Arguments = arguments,
            WorkingDirectory = workingDirectory,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8
        };

        // Ensure inner python calls use same venv
        try
        {
            var pythonDir = Path.GetDirectoryName(fileName);
            if (!string.IsNullOrEmpty(pythonDir))
            {
                var existingPath = psi.EnvironmentVariables["PATH"];
                psi.EnvironmentVariables["PATH"] = string.IsNullOrEmpty(existingPath)
                    ? pythonDir
                    : pythonDir + Path.PathSeparator + existingPath;

                var venvRoot = Directory.GetParent(pythonDir)?.FullName;
                if (!string.IsNullOrEmpty(venvRoot))
                    psi.EnvironmentVariables["VIRTUAL_ENV"] = venvRoot;
            }
        }
        catch (Exception ex)
        {
            AppendLog($"[WARN] Failed to adjust PATH for venv: {ex.Message}\n");
        }

        using var process = new Process { StartInfo = psi, EnableRaisingEvents = true };

        process.OutputDataReceived += (_, e) =>
        {
            if (e.Data != null)
                AppendLog(e.Data + "\n");
        };

        process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data != null)
                AppendLog("[ERR] " + e.Data + "\n");
        };

        try
        {
            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            await Task.Run(process.WaitForExit);

            AppendLog($"\nProcess exited with code {process.ExitCode}\n");
            ClearBusy($"Finished: {label} (exit {process.ExitCode})");
        }
        catch (Exception ex)
        {
            AppendLog($"\nEXCEPTION: {ex.Message}\n");
            ClearBusy("Idle (exception)");
        }
    }

    // ---------------------- UI helpers ----------------------

    private void AppendLog(string text)
    {
        Dispatcher.UIThread.Post(() =>
        {
            var stamp = DateTime.Now.ToString("HH:mm:ss ");
            LogTextBox.Text += stamp + text;

            if (!string.IsNullOrEmpty(LogTextBox.Text))
            {
                LogTextBox.CaretIndex = LogTextBox.Text.Length;
            }
        });
    }

    private void ClearLog()
    {
        LogTextBox.Text = "";
    }

    private void SetBusy(string status)
    {
        _isBusy = true;
        StatusTextBlock.Text = status;
        BusyProgressBar.IsVisible = true;

        RunBatchButton.IsEnabled = false;
        ReviewPendingButton.IsEnabled = false;
        ExportMidiButton.IsEnabled = false;

        BrowsePythonButton.IsEnabled = false;
        BrowseRepoButton.IsEnabled = false;
        BrowseRawButton.IsEnabled = false;
        BrowseExportButton.IsEnabled = false;
        BrowseSplitExportButton.IsEnabled = false; // ✅ new

    }

    private void ClearBusy(string status)
    {
        _isBusy = false;
        StatusTextBlock.Text = status;
        BusyProgressBar.IsVisible = false;

        RunBatchButton.IsEnabled = true;
        ReviewPendingButton.IsEnabled = true;
        ExportMidiButton.IsEnabled = true;

        BrowsePythonButton.IsEnabled = true;
        BrowseRepoButton.IsEnabled = true;
        BrowseRawButton.IsEnabled = true;
        BrowseExportButton.IsEnabled = true;
        BrowseSplitExportButton.IsEnabled = true; // ✅ new
        ExportMidiButton.IsEnabled = _hasRunBatch;

    }

    // ---------------------- MIDI splitting ----------------------

    private async Task SplitAllExportedMidisAsync(string pythonExe, string repoRoot, string combinedExportDir,
        string splitOutDir)
    {
        var midis = Directory.EnumerateFiles(combinedExportDir, "*.mid", SearchOption.TopDirectoryOnly).ToList();
        if (midis.Count == 0)
        {
            AppendLog($"[WARN] No .mid files found in: {combinedExportDir}\n");
            return;
        }

        foreach (var midi in midis)
        {
            var baseName = Path.GetFileNameWithoutExtension(midi);
            var songOut = Path.Combine(splitOutDir, baseName);

            Directory.CreateDirectory(songOut);

            var args = $"utils/split_midi_tracks.py \"{midi}\" --out \"{songOut}\"";
            await RunProcessAsync(pythonExe, args, repoRoot, "split-midi-tracks");
        }
    }
}



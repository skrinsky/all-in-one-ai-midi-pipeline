using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Threading;

namespace AiMidiGui;

public partial class MainWindow : Window
{
    private bool _isBusy;

    public MainWindow()
    {
        InitializeComponent();

        // Wire up buttons
        RunBatchButton.Click += RunBatchButton_OnClick;
        ReviewPendingButton.Click += ReviewPendingButton_OnClick;
        ExportMidiButton.Click += ExportMidiButton_OnClick;

        BrowsePythonButton.Click += BrowsePythonButton_OnClick;
        BrowseRepoButton.Click += BrowseRepoButton_OnClick;
        BrowseRawButton.Click += BrowseRawButton_OnClick;
        BrowseExportButton.Click += BrowseExportButton_OnClick;

        ClearLogButton.Click += ClearLogButton_OnClick;

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
        if (_isBusy) return;

        ClearLog();
        SetBusy("Running: export-midi …");

        string pythonExe = PythonPathTextBox.Text?.Trim() ?? "";
        string repoRoot = RepoRootTextBox.Text?.Trim() ?? "";
        string exportFolder = ExportFolderTextBox.Text?.Trim() ?? "out_midis";

        if (!File.Exists(pythonExe) || !Directory.Exists(repoRoot))
        {
            AppendLog("ERROR: Check Python path and repo root.\n");
            ClearBusy("Idle (error)");
            return;
        }

        // Make sure export folder exists (relative to repo root)
        string fullExportPath = Path.IsPathRooted(exportFolder)
            ? exportFolder
            : Path.Combine(repoRoot, exportFolder);

        Directory.CreateDirectory(fullExportPath);

        string args = $"pipeline.py export-midi --out \"{exportFolder}\"";
        await RunProcessAsync(pythonExe, args, repoRoot, "export-midi");
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
                // Store relative to repo root if possible
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

    // 🔧 Make sure any inner "python -m demucs.separate" calls
    // use the SAME venv python by putting its bin folder at the
    // front of PATH.
    try
    {
        var pythonDir = Path.GetDirectoryName(fileName);
        if (!string.IsNullOrEmpty(pythonDir))
        {
            var existingPath = psi.EnvironmentVariables["PATH"];
            if (string.IsNullOrEmpty(existingPath))
            {
                psi.EnvironmentVariables["PATH"] = pythonDir;
            }
            else
            {
                psi.EnvironmentVariables["PATH"] =
                    pythonDir + Path.PathSeparator + existingPath;
            }

            // Optional: advertise the virtual env to Python code
            var venvRoot = Directory.GetParent(pythonDir)?.FullName;
            if (!string.IsNullOrEmpty(venvRoot))
            {
                psi.EnvironmentVariables["VIRTUAL_ENV"] = venvRoot;
            }
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

            // Move caret to the end so the newest text is the “active” point
            if (!string.IsNullOrEmpty(LogTextBox.Text))
            {
                LogTextBox.CaretIndex = LogTextBox.Text.Length;
                // Avalonia will keep the caret in view without ScrollToEnd()
            }
        });
    }


    private async Task RunPipelineAsync(string args, string label)
    {
        SetBusy($"Running: {label}");

        RunBatchButton.IsEnabled = false;
        ReviewPendingButton.IsEnabled = false;
        ExportMidiButton.IsEnabled = false;

        try
        {
            await RunProcessAsync(
                PythonPathTextBox.Text,
                args,
                RepoRootTextBox.Text,
                label);
        }
        finally
        {
            RunBatchButton.IsEnabled = true;
            ReviewPendingButton.IsEnabled = true;
            ExportMidiButton.IsEnabled = true;
        }
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
    }
}


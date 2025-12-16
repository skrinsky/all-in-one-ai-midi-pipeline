using Avalonia;

namespace AiMidiGui
{
    internal sealed class Program
    {
        // Entry point
        public static void Main(string[] args) =>
            BuildAvaloniaApp()
                .StartWithClassicDesktopLifetime(args);

        public static AppBuilder BuildAvaloniaApp() =>
            AppBuilder
                .Configure<App>()
                .UsePlatformDetect()
                .LogToTrace();
    }
}
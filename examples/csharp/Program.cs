using System;
using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
public struct MiniNsfConfig
{
    public float source_sample_rate;
    public int upsample;
}

public static partial class MiniNsf
{
    private const string LibraryName = "mininsf";

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void mininsf_default_config(out MiniNsfConfig config);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int mininsf_fastsinegen_fast_f32(
        float[] f0,
        long batch,
        long n_frames,
        ref MiniNsfConfig config,
        float[] output
    );
}

public static class Program
{
    public static int Main()
    {
        MiniNsf.mininsf_default_config(out var config);

        float[] f0 = { 220.0f, 240.0f, 260.0f, 280.0f };
        float[] source = new float[f0.Length * config.upsample];

        int ret = MiniNsf.mininsf_fastsinegen_fast_f32(
            f0,
            1,
            f0.Length,
            ref config,
            source
        );
        if (ret != 0)
        {
            Console.Error.WriteLine($"mininsf failed: {ret}");
            return 1;
        }

        for (int i = 0; i < Math.Min(16, source.Length); i++)
        {
            Console.WriteLine($"{i} {source[i]:F9}");
        }
        return 0;
    }
}

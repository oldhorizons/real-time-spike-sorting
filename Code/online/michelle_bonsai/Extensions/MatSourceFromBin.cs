using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using OpenCV.Net;
using System.IO;
using System.Runtime.InteropServices;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Source)]
public class MatSourceFromBin
{
    [Description("The .dat or .bin filepath")]
    [Category("Configuration")]
    public string FilePath { get; set; }
    // C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/sim_hybrid_ZFM_10sec/continuous.bin

    [Description("The number of samples collected for each channel")]
    [Category("Configuration")]
    public int BufferSize { get; set; }
    // 30

    [Description("The number of channels in the data. defaults to 385 (number of channels in a synthesised recording)")]
    [Category("Configuration")]
    public int NumChannels { get; set; }
    // 385

    private Mat GetMat(BinaryReader reader)
    {
        byte[] buffer = reader.ReadBytes(NumChannels * BufferSize * 2);
        if (buffer.Length != NumChannels * BufferSize * 2) {
            return null;
        }
        Mat mat = new Mat(
            size: new Size(NumChannels, BufferSize),
            depth: Depth.U16,
            channels: NumChannels,
            data: Marshal.UnsafeAddrOfPinnedArrayElement(buffer, 0),
            step: NumChannels * 2
        );
        return mat;
    }

    // code snippet adapted from https://stackoverflow.com/questions/14454766/what-is-the-proper-way-to-create-an-observable-which-reads-a-stream-to-the-end, updated for BinaryReader
    private IObservable<Mat> GetObservable()
    {
        return Observable.Using(
            () => new BinaryReader(File.Open(FilePath, FileMode.Open)),
            reader =>  Observable.Return(GetMat(reader))
                                .Repeat()
                                .TakeWhile(data => data != null));
    }

    public IObservable<Mat> Process()
    {
        return GetObservable();
    }
}

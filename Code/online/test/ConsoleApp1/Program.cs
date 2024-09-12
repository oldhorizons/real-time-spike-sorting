// See https://aka.ms/new-console-template for more information

using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.IO;
using OpenCV.Net;
using System.Runtime.InteropServices;
using System.Reactive.Disposables;
using Bonsai.Reactive;

Console.WriteLine("Hello, World!");

SourceFromBin s = new SourceFromBin{
    FilePath = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/sim_hybrid_ZFM_10sec/continuous.bin",
    BufferSize = 30,
    NumChannels = 385,
    Frequency = 30000
};
// foreach (Mat m in s.Process().Next()) {
//     Console.WriteLine("Heyyyyyyy ;)");
//     Console.WriteLine(m.Data);
// }
Console.WriteLine(await s.Process().Count());

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Source)]
public class SourceFromBin
{
    [Description("The .dat or .bin filepath")]
    [Category("Configuration")]
    public string FilePath { get; set; }

    [Description("The number of samples collected for each channel")]
    [Category("Configuration")]
    public int BufferSize { get; set; }

    [Description("The number of channels in the data. defaults to 385 (number of channels in a synthesised recording)")]
    [Category("Configuration")]
    public int NumChannels { get; set; }

    [Description("The frequency of the sampler, in Hz")]
    [Category("Configuration")]
    public double Frequency { get; set; }

    private Mat GetMat(BinaryReader reader)
    {
        byte[] buffer = reader.ReadBytes(NumChannels * BufferSize * 2);
        if (buffer.Length != NumChannels * BufferSize * 2) {
            return null;
        }
        Mat mat = Mat.CreateMatHeader(buffer, BufferSize, NumChannels, Depth.U16, NumChannels);
        return mat;
    }

    // code snippet adapted from https://stackoverflow.com/questions/14454766/what-is-the-proper-way-to-create-an-observable-which-reads-a-stream-to-the-end, updated for BinaryReader
    private IObservable<Mat> GetObservable()
    {
        return Observable.Create<Mat>(async observer => {
            using (BinaryReader reader = new BinaryReader(File.Open(FilePath, FileMode.Open))) {
                while (true) {
                    // DateTime timeForNextStep = DateTime.Now.AddMilliseconds(BufferSize * 1_0000 / Frequency);
                    Mat mat = GetMat(reader);
                    if (mat == null) {
                        observer.OnCompleted();
                        break;
                    }
                    observer.OnNext(mat);
                    // TimeSpan delay = timeForNextStep - DateTime.Now;
                    // if (delay > TimeSpan.Zero) {
                    //     await Task.Delay(delay);
                    // }
                    await Task.Delay(TimeSpan.FromMilliseconds(10));
                    // Thread.Sleep(1);
                    // Console.WriteLine(DateTime.Now);
                }
            }
            return Disposable.Empty;
        });
    }


    public IObservable<Mat> Process()
    {
        return GetObservable();
    }
}



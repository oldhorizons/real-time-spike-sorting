using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.IO;
using OpenCV.Net;
using System.Runtime.InteropServices;
using System.Reactive.Disposables;
using System.Threading.Tasks;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Source)]
public class SourceFromBin
{
    [Description("The .dat or .bin filepath")]
    [Category("Configuration")]
    public string FilePath { get; set; }
    // C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/sim_hybrid_ZFM_10sec/continuous.bin

    [Description("The number of samples collected for each channel per sample")]
    [Category("Configuration")]
    public int BufferSize { get; set; }
    // 30

    [Description("The number of channels in the data. defaults to 385 (number of channels in a synthesised recording)")]
    [Category("Configuration")]
    public int NumChannels { get; set; }
    // 385

    [Description("The frequency of the sampler, in Hz")]
    [Category("Configuration")]
    public double Frequency { get; set; }
    // 30000

    private Mat GetMat(BinaryReader reader)
    {
        byte[] buffer = reader.ReadBytes(NumChannels * BufferSize * 2);
        if (buffer.Length != NumChannels * BufferSize * 2) {
            Console.Beep();
            Console.WriteLine("FINISHED");
            return null;
        }
        Mat mat = Mat.CreateMatHeader(buffer, NumChannels, BufferSize, Depth.S16, 1); //this was S32 but the actual data is S16 and I'm the silliest of geeses
        return mat;
    }

    private IObservable<Mat> GetObservable()
    {
        return Observable.Create<Mat>(async observer => {
            using (BinaryReader reader = new BinaryReader(File.Open(FilePath, FileMode.Open))) {
                while (true) {
                    Mat mat = GetMat(reader);
                    if (mat == null) {
                        observer.OnCompleted();
                        break;
                    }
                    observer.OnNext(mat);
                    // await Task.Delay(TimeSpan.FromMilliseconds(10));
                }
            }
            return Disposable.Empty;
        }); //.Delay(TimeSpan.FromMilliseconds(1000)) // BufferSize * 1000 / Frequency // there seems to be an inherent lower limit to delay times - about 64 times/second
    }

    public IObservable<Mat> Process()
    {
        return GetObservable();
    }
}

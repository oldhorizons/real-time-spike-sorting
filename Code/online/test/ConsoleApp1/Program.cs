// See https://aka.ms/new-console-template for more information

using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using OpenCV.Net;
using System.IO;
using System.Runtime.InteropServices;
using System.Reactive.Disposables;
using Bonsai.Reactive;

Console.WriteLine("Hello, World!");

LoadSpikeTemplates s = new LoadSpikeTemplates{
    SourcePath = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/sim_hybrid_ZFM_10sec/kilosort4/templates",
    BufferSize = 30,
    NumChannels = 385,
    Frequency = 30000
};

Console.WriteLine(await s.Process().Count());

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LoadSpikeTemplates
{
    [Description("The path to the folder containing the template .csv files")]
    [Category("Configuration")]
    public required string Sourcepath { get; set; }

    [Description("Target templates")]
    [Category("Configuration")]
    public int[] TemplatesToTrack { get; set; }

    public IObservable<int> Process()
    {
        return Observable.Return(0);
    }

    private Mat GetMat(string filename)
    {
        using(StreamReader reader = new StreamReader(filename))
        {
            List<List<float>> floats = new List<List<float>>();
            while (!reader.EndOfStream)
            {
                string[] line = reader.ReadLine().Split(',');
                List<float> floatLine = new List<float>();
                foreach(string item in line) 
                {
                    floatLine.Add(float.Parse(item));
                }
                floats.Add(floatLine);

                
                Mat mat = Mat.CreateMatHeader(buffer, NumChannels, BufferSize, Depth.S32, 1); //todo change back to S32 WORKS WITH U8

            }
        }
    }
}

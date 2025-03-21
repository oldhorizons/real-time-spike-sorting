using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using OpenCV.Net;
using System.IO;
using Bonsai.Dsp;

[Combinator]
[Description("Loads the spike templates into a handy dandy spike template format")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LoadSpikeTemplates
{
    [Description("The path to the folder containing the template .csv files")]
    [Category("Configuration")]
    public string SourcePath { get; set; }

    [Description("Target templates")]
    [Editor("Bonsai.Dsp.Design.SelectChannelEditor, Bonsai.Dsp.Design", "System.Drawing.Design.UITypeEditor, System.Drawing, Version=4.0.0.0")]
    [Category("Configuration")]
    public int[] TemplatesToTrack { get; set; }

    [Description("Target templates")]
    [Category("Configuration")]
    public int NumSamples { get; set; } //61

    [Description("whether to convert the data in F32 format to UINT8 with scaling (to match data feed bit depth)")]
    [Category("Configuration")]
    public bool ConvertToU8 {get; set; }

    public IObservable<SpikeWaveformCollection> Process()
    { 
        List<SpikeWaveform> templates = new List<SpikeWaveform>();
        // int numTemplates = TemplatesToTrack.Length;
        // byte[] templates = new byte[NumSamples * 4 * numTemplates];
        for (int i = 0; i < TemplatesToTrack.Length; i++) {
            string filename = String.Format("{0}/t{1}.csv", SourcePath, TemplatesToTrack[i]);
            SpikeWaveform template = GetSingleChanWaveform(filename);
            templates.Add(template);
        }
        SpikeWaveformCollection templateCollection = new SpikeWaveformCollection(
            templates, new Size(NumSamples,  TemplatesToTrack.Length));

        return Observable.Return(templateCollection);
    }

    // Returns the matrix representation of all templates
    // as a single Mat, where each row in the mat file is the max. channel of the template
    private SpikeWaveform GetSingleChanWaveform(string filename)
    {
        int numChannels = 0;
        float[] chanMax = new float[0];
        int[] chanMaxIndex = new int[0];
        List<List<float>> floats = new List<List<float>>();

        using(StreamReader reader = new StreamReader(filename))
        {
            int j = 0;
            //read list out
            while (!reader.EndOfStream)
            {
                // read line and parse to float array
                string[] line = reader.ReadLine().Split(',');
                List<float> floatLine = new List<float>();
                foreach(string item in line) {
                    floatLine.Add(float.Parse(item));
                }
                // count num samples, initialise numChannels and chanMax if this is the first run through
                if(numChannels == 0) {
                    numChannels = floatLine.ToArray().Length;
                    chanMax = new float[numChannels];
                    chanMaxIndex = new int[numChannels];
                }
                //update channel maximums
                for (int i = 0; i < numChannels; i++) {
                    if (floatLine[i] > chanMax[i]) {
                        chanMaxIndex[i] = j;
                    }
                    chanMax[i] = Math.Max(Math.Abs(floatLine[i]), chanMax[i]);
                }
                floats.Add(floatLine);
                j++;
            }
        }
        float maxValue = chanMax.Max();
        int maxIndex = chanMax.ToList().IndexOf(maxValue);

        //TODO: add an option to track multiple channels for this
        byte[] buffer = new byte[NumSamples * 4];
        System.Buffer.BlockCopy(floats[maxIndex].ToArray(),  0, buffer, 0, NumSamples * 4);
        Mat waveform = Mat.CreateMatHeader(buffer, 1, NumSamples, Depth.S32, 1);

        if (ConvertToU8) {
            Mat mat2 = new Mat(waveform.Rows, waveform.Cols, Depth.U8, 1);
            CV.ConvertScaleAbs(waveform, mat2, 0.0000425);
            waveform = mat2;
        }

        // return waveform;

        return new SpikeWaveform{
            ChannelIndex = maxIndex,
            SampleIndex = 0,
            Waveform = waveform,
        };
    }
}
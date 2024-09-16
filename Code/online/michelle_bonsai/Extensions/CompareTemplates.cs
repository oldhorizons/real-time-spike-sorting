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
[Description("Loads templates (todo clean this up) and compares them to the given channels")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class CompareTemplates
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
    public bool ConvertToU8 { get; set; }

    [Description("Confidence required to accept a spike as matched (-1 to 1)")]
    [Category("Configuration")]
    public float SimilarityThreshold { get; set; }

    public IObservable<float> Process(IObservable<SpikeWaveformCollection> source)
    {
        SpikeWaveformCollection templates = LoadTemplates();
        return Observable.Create<int>(observer => {
            return source.Subscribe(AlignWaveforms => {
                foreach (SpikeWaveform waveform in source.waveforms) {
                    foreach (SpikeWaveform template in templates) {
                        observer.OnNext(CosineSimilarity(waveform, templates));
                    }
        }})});
    }

    private float CosineSimilarity(SpikeWaveform source, SpikeWaveform template) {
        if (Math.Abs(source.ChannelIndex - template.ChannelIndex)) > SimilarityThreshold) {
            return -1;
        }
        // alt methods: 
        // euclidean distance EW YUCK NO THANK YOU
        // align, then cosine similarity (O(N)) - x dot y / magnitude(x) * magnitude(y) -> scale-invariant so robust to differences in amplitude
        // cross-correlation - reasonably fast - argmax() - needs FFT methods
        // Dynamic time warp[ing]
        // https://stackoverflow.com/questions/20644599/similarity-between-two-signals-looking-for-simple-measure
        AlignWaveforms(source, template);
        float dotProduct = source.waveform.Zip(template.waveform, (d1, d2) => d1 * d2)
                        .Sum();
        float sMag, tMag;
        sMag = tMag = 0;
        foreach (int value in template.waveform) {
            tMag += value * value;
        }
        foreach (int value in source.waveform) {
            sMag += value * value;
        }
        tMag = Math.Sqrt(tMag);
        sMag = Math.Sqrt(sMag);
        return (dotProduct / (tMag * sMag));
    }

    private void AlignWaveforms(ref SpikeWaveform source, ref SpikeWaveform template) {
        //initialise stuff
        int sMax, tMax, sMin, tMin, sMaxI, tMaxI, sMinI, tMinI;
        Mat sWav = source.waveform;
        Mat tWav = template.waveform;
        int lenDiff = sWav.Length - tWav.Length;
        sMaxI = tMaxI = sMinI = tMinI = 0;
        sMax = sMin = sWav[0];
        tMax = tMin = tWav[0];
        for (int i = 0; i < sWav.Length; i++) {
            if (sWav[i] > sMax) {
                sMax = sWav[i];
                sMaxI = i;
            }
            if (sWav[i] < sMin) {
                sMin = sWav[i];
                sMinI = i;
            }
        }
        for (int i = 0; i < tWav.Length; i++) {
            if (tWav[i] > tMax) {
                tMax = tWav[i];
                tMaxI = i;
            }
            if (tWav[i] < tMin) {
                tMin = tWav[i];
                tMinI = i;
            }
        }
        //todo because both are uint8_t the min won't matter but it could be handy for later so I'm keeping it?????
        int shift = sMaxI - tMaxI;
        lenDiff += shift;
        if (shift > 0) {
            // cut first [shift] off start of s
            sWav = sWav[shift..];
        } else if (shift < 0) {
            // cut first [-shift] off start of t
            tWav = tWav[-shift..];
        }
        if (lenDiff > 0) {
            // cut last diff off t
            tWav = tWav[..^lenDiff];
        } else if (lenDiff < 0) {
            // cut the last diff off s
            sWav = sWav[..^-lenDiff];
        }
        //update spike waveforms
        source.waveform = sWav;
        template.waveform = tWav;
    }

    private SpikeWaveformCollection LoadTemplates()
    { 
        List<SpikeWaveform> templates = new List<SpikeWaveform>();
        for (int i = 0; i < TemplatesToTrack.Length; i++) {
            string filename = String.Format("{0}/t{1}.csv", SourcePath, TemplatesToTrack[i]);
            SpikeWaveform template = GetSingleChanWaveform(filename);
            templates.Add(template);
        }
        SpikeWaveformCollection templateCollection = new SpikeWaveformCollection(
            templates, new Size(NumSamples,  TemplatesToTrack.Length));

        return templateCollection;
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

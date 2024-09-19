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
    // alt methods: 
    // euclidean distance EW YUCK NO THANK YOU
    // align, then cosine similarity (O(N)) - x dot y / magnitude(x) * magnitude(y) -> scale-invariant so robust to differences in amplitude
    // cross-correlation - reasonably fast - argmax() - needs FFT methods
    // Dynamic time warp[ing]
    // https://stackoverflow.com/questions/20644599/similarity-between-two-signals-looking-for-simple-measure
    public enum ComparisonMethod {
        Cosine,
        CrossCor,
        Dtw
    }

    [Description("The method to use to compare")]
    [Category("Configuration")]
    public ComparisonMethod comparisonMethod;

    [Description("The path to the folder containing the template .csv files")]
    [Category("Configuration")]
    public string SourcePath { get; set; }

    [Description("Target templates")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    [Editor("Bonsai.Dsp.Design.SelectChannelEditor, Bonsai.Dsp.Design", "System.Drawing.Design.UITypeEditor, System.Drawing, Version=4.0.0.0")]
    [Category("Configuration")]
    public int[] TemplatesToTrack { get; set; }

    [Description("Target templates")]
    [Category("Configuration")]
    public int NumSamples { get; set; } //61

    [Description("whether to convert the data in F32 format to UINT8 with scaling (to match data feed bit depth)")]
    [Category("Configuration")]
    public bool ConvertToU8 { get; set; }

    [Description("Channel closeness required to accept a spike as matched")]
    [Category("Configuration")]
    public float DistanceThreshold { get; set; }

    [Description("Confidence required to accept a spike as matched (-1 to 1)")]
    [Category("Configuration")]
    public float SimilarityThreshold { get; set; }

    // so I can output more than one thing, for visualisation purposes.
    public class SpikeComparer {
        public Mat template {get; set;}
        public Mat source {get; set;}
        public float similarity {get; set;}
        public int accepted {get; set;}
    }

    public IObservable<SpikeComparer> Process(IObservable<SpikeWaveformCollection> source)
    {
        List<SpikeTemplate> templates = LoadTemplates();
        Console.WriteLine("TEMPLATES LOADED");
        return Observable.Create<SpikeComparer>(observer => {
            return source.Subscribe(waveforms => {
                foreach (SpikeWaveform waveform in waveforms) {
                    foreach (SpikeTemplate template in templates) {
                        observer.OnNext(
                            new SpikeComparer() {
                                template = template.Waveform,
                                source = waveform.Waveform,
                                similarity = SimilarityMeasure(waveform.Waveform, template),
                        });
                    }
                }
            });
        });
    }

    private float SimilarityMeasure(Mat source, SpikeTemplate template) {
        //DO NOT pass un-cloned template waveforms through any of the functions below.
        return CosineSimilarity(source, template.Waveform.Clone(), template.AlignMax);
        // switch (comparisonMethod) {
        //     case (ComparisonMethod.Cosine) {
        //         return CosineSimilarity(source, template);
        //     }
        // }
    }

    private float CosineSimilarity(Mat sWav, Mat tWav, bool alignMax) {
        // if (Math.Abs(source.ChannelIndex - template.ChannelIndex) > DistanceThreshold) {
        //     return -1f;
        // }
        AlignWaveforms(ref sWav, ref tWav, alignMax);
        int max = sWav.Cols;
        Console.WriteLine("Num Cols: {0}", max);
        double dotProduct = 0;
        double sMag, tMag;
        sMag = tMag = 0;
        Mat halfDot = sWav * tWav;
        sWav *= sWav;
        tWav *= tWav;
        for (int i = 0; i < max; i++) {
            dotProduct += halfDot[i].Val0;
            tMag += tWav[i].Val0;
            sMag += sWav[i].Val0;
        }
        tMag = Math.Sqrt(tMag);
        sMag = Math.Sqrt(sMag);
        float cosineSimilarity = (float)(dotProduct / (tMag * sMag));
        Console.WriteLine("dotProduct: {0}, tMag: {1}, sMag: {2}, cosineSim: {3}", 
                dotProduct, tMag, sMag, cosineSimilarity);
        return cosineSimilarity;
    }

    private void AlignWaveforms(ref Mat sWav, ref Mat tWav, bool alignMax) {
        //initialise stuff
        double sMax, tMax, sMin, tMin;
        int sMaxI, tMaxI, sMinI, tMinI;
        int lenDiff = sWav.Cols - tWav.Cols;
        sMaxI = tMaxI = sMinI = tMinI = 0;
        sMax = sMin = sWav[0].Val0;
        tMax = tMin = tWav[0].Val0;
        for (int i = 0; i < sWav.Cols; i++) {
            if (sWav[i].Val0 > sMax) {
                sMax = sWav[i].Val0;
                sMaxI = i;
            } 
            else if (sWav[i].Val0 < sMin) {
                sMin = sWav[i].Val0;
                sMinI = i;
            }
        }
        for (int i = 0; i < tWav.Cols; i++) {
            if (tWav[i].Val0 > tMax) {
                tMax = tWav[i].Val0;
                tMaxI = i;
            } 
            else if (tWav[i].Val0 < tMin) {
                tMin = tWav[i].Val0;
                tMinI = i;
            }
        }
        //align either highest or lowest point, depending on template
        int offset = alignMax ? sMaxI - tMaxI : sMinI - tMinI;
        lenDiff -= offset;
        int sOff = offset > 0 ? offset : 0;
        int tOff = offset < 0 ? -offset : 0;
        int sCut = lenDiff > 0 ? lenDiff : 0;
        int tCut = lenDiff < 0 ? -lenDiff : 0;
        Console.WriteLine("lenDiff: {0}, offset: {1}", lenDiff, offset);
        Console.WriteLine("sCols: {0}, sMaxI: {1}, sOff: {2}, sCut: {3} | tCols: {4}, tMaxI: {5}, tOff: {6}, tCut: {7}",
                            sWav.Cols, sMaxI, sOff, sCut, tWav.Cols, tMaxI, tOff, tCut);

        sWav = ShortenMat(sWav, sOff, sCut);
        tWav = ShortenMat(tWav, tOff, tCut);
    }
    
    private Mat ShortenMat(Mat source, int start, int end) {
        int newLen = source.Cols - (start + end);
        if (newLen == source.Cols) {
            return source;
        }
        double[] newWav = new double[newLen];
        for (int i = 0; i < newLen; i++) {
            newWav[i] = source[i + start].Val0;
        }
        return Mat.CreateMatHeader(newWav);
    }

    private List<SpikeTemplate> LoadTemplates()
    { 
        List<SpikeTemplate> templates = new List<SpikeTemplate>();
        for (int i = 0; i < TemplatesToTrack.Length; i++) {
            string filename = String.Format("{0}/t{1}.txt", SourcePath, TemplatesToTrack[i]); //TODO CSV
            SpikeTemplate template = GetSingleChanWaveform(filename);
            templates.Add(template);
        }
        return templates;
    }

    // Returns the matrix representation of a template as a 1-d Mat file
    // todo add a way to track multiple channels? 
    private SpikeTemplate GetSingleChanWaveform(string filename)
    {
        int numChannels = 0;
        float[] chanMax = new float[1];
        int[] chanMaxIndex = new int[1];
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
                    if (Math.Abs(floatLine[i]) > chanMax[i]) {
                        chanMaxIndex[i] = j;
                        chanMax[i] = Math.Abs(floatLine[i]);
                    }
                }
                floats.Add(floatLine);
                j++;
            }
        }
        float maxValue = chanMax.Max();
        int maxIndex = chanMax.ToList().IndexOf(maxValue);


        // cut off the leading and trailing zeroes because you don't want to try to match a flat line
        int offset = 0;
        int end = NumSamples;
        for (int i = 0; i < NumSamples; i++) {
            if (floats[maxIndex][i] != 0) {
                break;
            } else {
                offset++;
            }
        }
        for (int i = NumSamples; i > 0; i--) {
            if (floats[maxIndex][i] != 0) {
                break;
            } else {
                end--;
            }
        }

        //add a LITTLE buffer of zeroes
        offset = Math.Max(0, offset - 2);
        end = Math.Min(NumSamples, end + 2);

        // get the channel info at each sample
        float[] buffer = new float[NumSamples];
        for (int i = offset; i < end; i++) {
            buffer[i] = floats[maxIndex][i];
        }

        Mat waveform = Mat.FromArray(buffer);

        // return waveform
        return new SpikeTemplate{
            ChannelIndex = maxIndex,
            SampleIndex = 0,
            Waveform = waveform,
            AlignMax = buffer.Contains(maxValue)
        };
    }

    protected class SpikeTemplate : SpikeWaveform {
        public bool AlignMax {get; set;}
    }
}

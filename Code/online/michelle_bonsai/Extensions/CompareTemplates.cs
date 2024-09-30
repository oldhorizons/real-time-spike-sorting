using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using OpenCV.Net;
using System.IO;
using Bonsai.Dsp;
using System.Text;

[Combinator]
[Description("Loads templates (todo clean this up) and compares them to the given channels")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class CompareTemplates
{
    // alt methods: 
    // align, then cosine similarity (O(N)) - x dot y / magnitude(x) * magnitude(y) -> scale-invariant so robust to differences in amplitude
    // cross-correlation - reasonably fast - argmax() - needs FFT methods
    // https://stackoverflow.com/questions/20644599/similarity-between-two-signals-looking-for-simple-measure
    public enum ComparisonMethod {
        Cosine,
        DTW,
    }

    [Description("The method to use to compare")]
    [Category("Configuration")]
    public ComparisonMethod comparisonMethod { get; set; }

    [Description("The path to the folder containing data")]
    [Category("Configuration")]
    public string SourcePathStart { get; set; }

    [Description("Any source path suffixes")]
    [Category("Configuration")]
    public string SourcePathSuffix { get; set; }

    [Description("Target templates")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    [Editor("Bonsai.Dsp.Design.SelectChannelEditor, Bonsai.Dsp.Design", "System.Drawing.Design.UITypeEditor, System.Drawing, Version=4.0.0.0")]
    [Category("Configuration")]
    public int[] TemplatesToTrack { get; set; }

    [Description("Whether to add visualisation output. May increase latency.")]
    [Category("Configuration")]
    public bool VisualisationOutput {get; set;}

    // [Description("whether to convert the data in F32 format to UINT8 with scaling (to match data feed bit depth)")]
    // [Category("Configuration")]
    // public bool ConvertToU8 { get; set; }

    [Description("Channel closeness required to accept a spike as matched")]
    [Category("Configuration")]
    public float DistanceThreshold { get; set; }

    [Description("Confidence required to accept a spike as matched (-1 to 1)")]
    [Category("Configuration")]
    public float SimilarityThreshold { get; set; }

    // so I can output more than one thing, for visualisation purposes.
    public class SpikeComparer {
        public SpikeWaveformCollection TemplatesForVis {get; set;}
        public List<Mat> Templates {get; set;}
        public Mat Source {get; set;}
        public float[] Similarities {get; set;}
        public bool[] Accepted {get; set;}
        public String SimilarityMessage {get; set;}
    }

    public IObservable<SpikeComparer> Process(IObservable<SpikeWaveformCollection> source)
    {
        List<SpikeTemplate> templates = LoadTemplates();
        if (VisualisationOutput) {
            return ProcessForVisualisation(source, templates);
        } else {
            return ProcessForCLosedLoop(source, templates);
        }
    }

    private IObservable<SpikeComparer> ProcessForVisualisation(IObservable<SpikeWaveformCollection> source, List<SpikeTemplate> templates) {
        return Observable.Create<SpikeComparer>(observer => {
            return source.Subscribe(waveforms => {
                foreach (SpikeWaveform waveform in waveforms) {
                    StringBuilder similarityMessage = new StringBuilder();
                    SpikeComparer spikeComparer = new SpikeComparer() {
                        Source = waveform.Waveform,
                        TemplatesForVis  = new SpikeWaveformCollection(new Size(1, 60)),
                        Templates = new List<Mat>()
                    };
                    foreach (SpikeTemplate template in templates) {
                        spikeComparer.Templates.Add(template.Waveform);
                        spikeComparer.TemplatesForVis.Add(template);
                        similarityMessage.AppendFormat("CH{0} | t{1} tId{2}| tChan {3} | sim {4}\r\n", 
                                                        waveform.ChannelIndex,
                                                        template.Id, 
                                                        template.ChannelIndex,
                                                        template.SampleIndex, 
                                                        SimilarityMeasure(waveform, template));
                    }
                    spikeComparer.SimilarityMessage = similarityMessage.ToString();
                    observer.OnNext(spikeComparer);
                }
            });
        });
    }

    private IObservable<SpikeComparer> ProcessForCLosedLoop(IObservable<SpikeWaveformCollection> source, List<SpikeTemplate> templates) {
        return Observable.Create<SpikeComparer>(observer => {
            return source.Subscribe(waveforms => {
                foreach (SpikeWaveform waveform in waveforms) {
                    List<float> similarities = new List<float>();
                    SpikeComparer spikeComparer = new SpikeComparer();
                    foreach (SpikeTemplate template in templates) {
                        similarities.Add(SimilarityMeasure(waveform, template));
                    }
                    spikeComparer.Similarities = similarities.ToArray();
                    observer.OnNext(spikeComparer);
                }
            });
        });
    }

    private float SimilarityMeasure(SpikeWaveform source, SpikeTemplate template) {
        // first check distance tolerance
        if (Math.Abs(source.ChannelIndex - template.ChannelIndex) > DistanceThreshold) {
            return -1f;
        }
        //DO NOT pass un-cloned template waveforms through any of the functions below.
        if (comparisonMethod == ComparisonMethod.Cosine) {
            return CosineSimilarity(source.Waveform, template.Waveform.Clone(), template.AlignMax);
        } else if (comparisonMethod == ComparisonMethod.DTW) {
            return DynamicTimeWarp(source.Waveform, template.Waveform);
        } else {
            return CosineSimilarity(source.Waveform, template.Waveform.Clone(), template.AlignMax);
        }
    }

    private float CosineSimilarity(Mat sWav, Mat tWav, bool alignMax) {
        AlignWaveforms(ref sWav, ref tWav, alignMax);
        int max = sWav.Cols;
        // Console.WriteLine("Num Cols: {0}", max);
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
        return cosineSimilarity;
    }

    //  https://github.com/doblak/ndtw for some dope visualisations
    // gonna be real I took this algorithm off wikipedia https://en.wikipedia.org/wiki/Dynamic_time_warping
    // there are less naive implementations that allow for early stopping / pruning (e.g. https://arxiv.org/pdf/2102.05221) 
    // but I have other things I need to do first
    private float DynamicTimeWarp(Mat sWav, Mat tWav) {
        double[,] DTW  = new double[sWav.Cols,tWav.Cols];
        for (int i = 0; i < sWav.Cols; i++) {
            for (int j = 0; j < tWav.Cols; j++) {
                DTW[i,j] = double.MaxValue;
            }
        }
        DTW[0,0] = 0;
        for (int i = 1; i < sWav.Cols; i++) {
            for (int j = 1; j < tWav.Cols; j++) {
                double cost = euclideanDist(sWav[i].Val0, i, tWav[j].Val0, j);
                DTW[i,j] = cost + Math.Min(DTW[i-1, j],
                                Math.Min(DTW[i, j-1],
                                DTW[i-1,j-1]));
            }
        }
        return (float)DTW[sWav.Cols-1,tWav.Cols-1];
    }

    private double euclideanDist(double a, int ai, double b, int bi) {
        return Math.Sqrt((a-b)*(a-b) + (ai-bi)*(ai-bi));
    }

    //align waveforms by highest or lowest point and crop to same length
    //NB EDITS IN PLACE. DO NOT PASS WAVEFORMS HERE YOU DON'T WANT CHANGED
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
        //align either highest or lowest point
        int offset = alignMax ? sMaxI - tMaxI : sMinI - tMinI;
        lenDiff -= offset;
        int sOff = offset > 0 ? offset : 0;
        int tOff = offset < 0 ? -offset : 0;
        int sCut = lenDiff > 0 ? lenDiff : 0;
        int tCut = lenDiff < 0 ? -lenDiff : 0;

        sWav = ShortenMat(sWav, sOff, sCut);
        tWav = ShortenMat(tWav, tOff, tCut);
    }
    
    //crops a 1-dimensional Mat to within a given start and end index.
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
        StringBuilder consoleMessage = new StringBuilder();
        for (int i = 0; i < TemplatesToTrack.Length; i++) {
            string filename = String.Format("{0}{1}/kilosort4/templates/t{2}.csv", SourcePathStart, SourcePathSuffix, TemplatesToTrack[i]);
            SpikeTemplate template = GetSingleChanWaveform(filename, i);
            template.Id = TemplatesToTrack[i];
            templates.Add(template);
            consoleMessage.AppendFormat("T{0}C{1}ID{2} ", TemplatesToTrack[i], template.SampleIndex, template.ChannelIndex);
        }
        Console.WriteLine(consoleMessage.ToString());
        return templates;
    }

    // Returns the matrix representation of a template as a 1-d Mat file
    // todo add a way to track multiple channels? 
    private SpikeTemplate GetSingleChanWaveform(string filename, int idx)
    {
        int numChannels = 0;
        int numSamples = 0;
        float[] chanMax = new float[1];
        int[] chanMaxIndex = new int[1];
        List<List<float>> floats = new List<List<float>>();

        using(StreamReader reader = new StreamReader(filename))
        {
            //read list out
            while (!reader.EndOfStream)
            {
                // read line and parse to float array
                string[] line = reader.ReadLine().Split(',');
                List<float> floatLine = new List<float>();
                foreach(string item in line) {
                    floatLine.Add(float.Parse(item));
                }
                // initialise numChannels and chanMax if this is the first run through
                if(numChannels == 0) {
                    numChannels = floatLine.ToArray().Length;
                    chanMax = new float[numChannels];
                    chanMaxIndex = new int[numChannels];
                }
                //update channel maximums
                for (int i = 0; i < numChannels; i++) {
                    if (Math.Abs(floatLine[i]) > chanMax[i]) {
                        chanMaxIndex[i] = numSamples;
                        chanMax[i] = Math.Abs(floatLine[i]);
                    }
                }
                floats.Add(floatLine);
                numSamples++;
            }
        }
        float maxValue = chanMax.Max();
        int maxIndex = chanMax.ToList().IndexOf(maxValue);

        // get the channel info at each sample
        float[] buffer = new float[numSamples];
        for (int i = 0; i < numSamples; i++) {
            buffer[i] = floats[i][maxIndex];
        }

        Mat waveform = Mat.FromArray(buffer);

        // return waveform
        // NB to be CORRECT ChannelIndex should be maxIndex and SampleIndex should be 0
        // but to be VISIBLE it's easier if channelindex = idx
        return new SpikeTemplate{
            ChannelIndex = idx,
            SampleIndex = maxIndex,
            Waveform = waveform,
            AlignMax = buffer.Contains(maxValue)
        };
    }

    protected class SpikeTemplate : SpikeWaveform {
        public bool AlignMax {get; set;}
        public int Id {get; set;}
    }
}
